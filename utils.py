import torch
import torch.nn as nn
import torch.nn.functional as F
import re

import re
import copy
# from word2number import w2n

unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    
def norm_str2bool(s: str) -> bool | None:
    """Converts a string representation of a boolean value to its corresponding boolean value."""
    s = str(s).lower().strip().replace("noindent", "")
    if any(pos in s for pos in ["yes", "true"]):
        return True
    elif any(neg in s for neg in ["no", "false"]):
        return False
    else:
        return None
    
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")

    if len(splits) < 2:
        return new_string
    
    new_string = splits[0] 
    for split in splits[1:]:
        if split != '' and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _fix_sqrt_v2(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

def _fix_a_slash_b_v2(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
                     (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                     (',\\text{and}', ','), ('\\text{and}', ','),
                     ('\\text{m}', '\\text{}'), ('\\le', '<')]
    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
        'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
        'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f'
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(\\text\{)\((.*?)\)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    assert '\n' not in final_answer
    assert '\r' not in final_answer
    assert '\f' not in final_answer
    if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
        final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]

    if len(re.findall(r'answer?is:?(.*)', final_answer)) > 0:
        final_answer = re.findall(r'answer?is:?(.*)', final_answer)[-1]

    if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
        final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]

    if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
        final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
    final_answer = final_answer.strip()
    if 'rac' in final_answer and '\\frac' not in final_answer:
        final_answer = final_answer.replace('rac', '\\frac')

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def is_equiv(str1, str2, verbose=False):
    
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        if ss1 == ss2:
            return True
        ss1 = normalize_final_answer(ss1)
        ss2 = normalize_final_answer(ss2)
        if ss1 == ss2:
            return True
    except Exception:
        pass

    try:
        ss1 = normalize_final_answer(str1)
        ss2 = normalize_final_answer(str2)
        if ss1 == ss2:
            return True
    except Exception:
        pass

    return str1 == str2



def extract_answer(pred_str, data_name, use_last_number=False):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return "", None
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
        
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    og_pred = copy.deepcopy(pred)

    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    if pred != "":
        pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])

    if pred != '' and pred in pred_str:
        pred_start = pred_str.find(pred)
        pred_span = (pred_start, pred_start + len(pred))
    else:
        pred_span = None

    if og_pred != '' and og_pred in pred_str:
        of_pred_start = pred_str.find(og_pred)
        og_pred_span = (of_pred_start, of_pred_start + len(og_pred))
        if pred_span is None:
            pred_span = og_pred_span
    else:
        og_pred_span = None

    return (pred, pred_span), (og_pred, og_pred_span)

def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    # string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_sqrt_v2(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b_v2(string)

    return string


class BaseTask:
    INVALID_ANS = "[invalid]"
    def __init__(self, encode_format):
        self.encode_format = encode_format
        assert self.encode_format in ['instruct', 'qa']

    def encode_prompt(self, example):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def extract_gt_answer(self, completion):
        match = self.GT_ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.INVALID_ANS

    def extract_model_answer(self, completion):
        if self.encode_format == 'qa':
            completion = completion.split("\nA: ")[1].split("\nQ: ")[0].split("\nQuestion: ")[0].split("\n[Question]: ")[0].split("\nB: ")[0]

        matches = list(re.finditer(self.MODEL_ANS_RE, completion))
        if len(matches) > 0:
            match = matches[-1]
            return match.group().lower(), (match.start(), match.end())
        else:
            return self.INVALID_ANS, None

    def is_correct(self, gt_example, model_answer):
        gt_answer = self.extract_gt_answer(gt_example[self.gt_key])
        assert gt_answer != self.INVALID_ANS
        return model_answer == gt_answer
    
    def correct(self, gt_answer, model_answer):
        gt_answer = _strip_string(gt_answer)
        model_answer = _strip_string(model_answer)
        return is_equiv(model_answer,gt_answer)


class GSMTask(BaseTask):
    GT_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    MODEL_ANS_RE = re.compile(r"([-0-9][0-9\,\.]*[0-9])|([0-9])")

    def __init__(self, encode_format):
        super().__init__(encode_format)
        self.gt_key = "answer"

    def encode_prompt(self, example):
        if self.encode_format == 'instruct':
            return '[INST]{}[/INST]'.format(example['question'])
        elif self.encode_format == 'qa':
            return 'Q: {}\nA:'.format(example['question'])

class MathTask(BaseTask):

    def __init__(self, encode_format):
        super().__init__(encode_format)
        self.gt_key = "solution"

    def encode_prompt(self, example):
        if self.encode_format == 'instruct':
            return '[INST]{}[/INST]'.format(example['problem'])
        elif self.encode_format == 'qa':
            return 'Q: {}\nA:'.format(example['problem'])
        
    

    def extract_model_answer(self, completion):
        if self.encode_format == 'qa':
            completion = completion.split("\nQ: ")[0].split("\nQuestion: ")[0].split("\n[Question]: ")[0].split("\nB: ")[0].strip()
            resp_str = completion.split("\nA: ")[-1].strip()
        if "herefore" in resp_str:
            resp = resp_str.split("herefore")[-1].strip()

        if "box" in resp_str:
            resp = last_boxed_only_string(resp_str)
            resp = remove_boxed(resp)
        else:
            # should be answer only
            if "is the ans" in resp_str:
                resp = re.split(r"(,|\.|\!\|?)", resp_str.split("is the ans")[-2].strip())[
                    -1
                ].strip()
            elif "is our ans" in resp_str:
                resp = re.split(r"(,|\.|\!\|?)", resp_str.split("is our ans")[-2].strip())[
                    -1
                ].strip()
            elif "answer is" in resp_str:
                resp = resp_str.split("answer is")[-1].strip()
            elif "answer:" in resp_str:
                resp = resp_str.split("answer:")[-1].strip()
            elif "answer :" in resp_str:
                resp = resp_str.split("answer :")[-1].strip()
            elif "statement" in resp_str:
                resp = resp_str.split("is ")[-1].strip()
            else:
                resp = None
            
        #resp need to use as regex for complition
        if resp is not None:
            if resp.startswith("$") and resp.endswith("$"):
                resp = resp[1:-1]
            MODEL_ANS_RE = re.escape(r"{}".format(resp))
            matches = list(re.finditer(MODEL_ANS_RE, completion))
            match = matches[-1]
            bool_resp = norm_str2bool(resp)
            if bool_resp is not None:
                return str(bool_resp), (match.start(), match.end())
            return match.group(), (match.start(), match.end())
        
        else:
            LATEX_ANS_RE = re.compile(r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\(|\\\[)",re.DOTALL)
            matches = list(re.finditer(LATEX_ANS_RE, completion))
            if len(matches) > 0:
                match = matches[-1]
                answer = match.group()
                start_pos = match.start()
                end_pos = match.end()
                if '=' in answer:
                    split_position = answer.rfind('=') + 1
                    answer = answer[split_position:].strip()
                    # Calculate new positions for the extracted part
                    start_pos = start_pos + split_position
                    end_pos = end_pos
                if answer.startswith("$"):
                    answer = answer[1:]
                    start_pos = start_pos + 1
                if answer.endswith("$"):
                    answer = answer[:-1]
                    end_pos = end_pos - 1
                return answer, (start_pos, end_pos)
            

            NUM_ANS_RE = re.compile(r"-?\d*\.?\d+")
            matches = list(re.finditer(NUM_ANS_RE, completion))
            if len(matches) > 0:
                match = matches[-1]
                return match.group(), (match.start(), match.end())
            
            return self.INVALID_ANS, None
    def extract_gt_answer(self, completion):
        match = last_boxed_only_string(completion)
        if match:
            return remove_boxed(match)
        else:
            return self.INVALID_ANS
        
    def is_correct(self, gt_example, model_answer):
        gt_answer = self.extract_gt_answer(gt_example[self.gt_key])
        gt_answer = _strip_string(gt_answer)
        model_answer = _strip_string(model_answer)
        assert gt_answer != self.INVALID_ANS
        return gt_answer,is_equiv(model_answer,gt_answer)
    
    

class MathTask_v2(BaseTask):

    def __init__(self, encode_format):
        super().__init__(encode_format)

    def encode_prompt(self, example):
        if self.encode_format == 'instruct':
            return '[INST]{}[/INST]'.format(example['problem'])
        elif self.encode_format == 'qa':
            prompt_suffix = r'Please reason and then put your final answer within \boxed{}.'
            return '\n'.join([example['problem'], prompt_suffix])

    def extract_model_answer(self, completion):
        if self.encode_format == 'qa':
            completion = completion.split("\nQ: ")[0].split("\nQuestion: ")[0].split("\n[Question]: ")[0].split("\nB: ")[0].strip()

        (model_answer, span), (model_answer_v2, span_v2) = extract_answer(completion, 'math')
        if model_answer == "":
            return self.INVALID_ANS, None
        else:
            return model_answer, span

    def extract_gt_answer(self, completion):
        match = last_boxed_only_string(completion)
        if match:
            return remove_boxed(match)
        else:
            return self.INVALID_ANS
        
    def is_correct(self, gt_example, model_answer):
        gt_answer = gt_example['answer']
        assert gt_answer != self.INVALID_ANS
        return gt_answer, is_equiv(model_answer,gt_answer)


class MLPProbe_2(nn.Module):
    def __init__(self, d, hidden = 100):
        super().__init__()
        self.linear1 = nn.Linear(d, hidden)
        self.linear2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.linear1(x)
        h = F.relu(h)
        o = self.linear2(h)
        return torch.sigmoid(o)

