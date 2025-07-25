from method import *
from utils import *
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # setting up model
    parser.add_argument("--model_name", type=str, default="llama2-7b-chat", help="Name of the model to use")
    parser.add_argument("--device", type=int, default=None, help="Device to use for the model")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Name of the dataset to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--train_num_examples", type=int, default=7473, help="Number of examples to generate")
    parser.add_argument("--test_num_examples", type=int, default=1319, help="Number of examples to generate")
    parser.add_argument("--cot_n_branches", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--encode_format", type=str, default="qa", help='instruct or qa')
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to use (if not all layers)")
    
    # reward model training 
    parser.add_argument("--nepochs", type=int, default=100)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--hs_train_file_path", type=str, required=True)
    parser.add_argument("--model_answer_train_file_path", type=str, required=True)
    parser.add_argument("--gt_answer_train_file_path", type=str, required=True)

    parser.add_argument("--hs_test_file_path", type=str, required=True)
    parser.add_argument("--gt_answer_test_file_path", type=str, required=True)
    parser.add_argument("--model_answer_test_file_path", type=str, required=True)

    parser.add_argument("--exp_description", type=str, default=None)
    parser.add_argument("--aggregate", type=str, default='sum', help="How to aggregate the model answers, 'sum' or 'max'")

    parser.add_argument("--use_loss3_logic", action="store_true", help="Use logic-based loss3 for training")

    parser.add_argument("--probe_file", type=str, default=None, help="probe file load")

    args = parser.parse_args()

    return args


def main(args):

    task = GSMTask(args.encode_format)
    all_trainning_hs_data, all_training_model_answers, all_training_gt_answers, \
        all_validation_hs_data, all_validation_model_answers, all_validation_gt_answers = load_train_and_dev_data(args)
    
    print('all_trainning_hs_data', all_trainning_hs_data.shape)

    model = LOVER(
            all_train_hs=all_trainning_hs_data
            , all_train_answers=all_training_model_answers
            , all_dev_hs=all_validation_hs_data
            , all_dev_gt_answers=all_validation_gt_answers
            , all_dev_model_answers=all_validation_model_answers
            , nepochs=args.nepochs 
            , ntries=args.ntries
            , lr=args.lr
            , linear=args.linear
            , weight_decay=args.weight_decay 
            , device=f'cuda:{args.device}' if args.device is not None else 'cuda'
            , task=task
            , probe_file=args.probe_file
            , data_layer=args.layer
            , use_loss3_logic=args.use_loss3_logic
            , aggregate= args.aggregate
        )

    if model.probe_file is None:
        model.repeated_train() 

    all_hs_layer, all_test_model_answers, all_test_gt_response_answers = load_test_data(args)
    test_result = decode_with_rewad_model(task, args, model, all_test_gt_response_answers, all_test_model_answers, all_hs_layer)

    return test_result

if __name__ == '__main__':

    args = get_args()

    for k in args.__dict__:
        logging.info(k + ":\t" + str(args.__dict__[k]))

    test_result = main(args)

    store_result(args, test_result,  './result/')
    best_acc = test_result['best_acc']
    print(f'method best acc: {100 * best_acc:.2f}')
    print(f'data_layer:{args.layer}, lr:{args.lr}')
    print(f'epoches:{args.nepochs}, linear:{args.linear}')
