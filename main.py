from _ast import Expression
from copy import copy

import ply.lex as lex
import ply.yacc as yacc
import inspect

tokens = (
    'IDENTIFIER',
    'PROGRAM',
    'VAR',
    'INT',
    'REAL',
    'BEGIN',
    'END',
    'IF',
    'THEN',
    'ELSE',
    'WHILE',
    'DO',
    'PRINT',
    'SWITCH',
    'OF',
    'DONE',
    'DEFAULT',
    'SEMICOL',
    'COL',
    'COMMA',
    'ASSIGNMENT',
    'PLUS',
    'MINUS',
    'MUL',
    'DIVIDE',
    'MOD',
    'GT',
    'EQ',
    'LT',
    'NE',
    'LE',
    'GE',
    'AND',
    'OR',
    'NOT',
    'LPAREN',
    'RPAREN',
    'NUMBER',
    'REALNUMBER',
)

t_PLUS = r'\+'
t_MINUS = r'-'
t_MUL = r'\*'
t_DIVIDE = r'/'
t_GT = r'>'
t_NE = r'<>'
t_LT = r'<'
t_LE = r'<='
t_EQ = r'='
t_GE = r'>='
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COMMA = r','
t_COL = r':'
t_SEMICOL = r';'
t_ASSIGNMENT = r':='


def t_MOD(t):
    r'\bmod\b'
    return t


def t_AND(t):
    r'\band\b'
    return t


def t_OR(t):
    r'\bor\b'
    return t


def t_NOT(t):
    r'\bnot\b'
    return t


def t_INT(t):
    r'\bint\b'
    return t


def t_REAL(t):
    r'\breal\b'
    return t


def t_BEGIN(t):
    r'\bbegin\b'
    return t


def t_VAR(t):
    r'\bvar\b'
    return t


def t_END(t):
    r'\bend\b'
    return t


def t_IF(t):
    r'\bif\b'
    return t


def t_THEN(t):
    r'\bthen\b'
    return t


def t_ELSE(t):
    r'\belse\b'
    return t


def t_DO(t):
    r'\bdo\b'
    return t


def t_WHILE(t):
    r'\bwhile\b'
    return t


def t_PRINT(t):
    r'\bprint\b'
    return t


def t_SWITCH(t):
    r'\bswitch\b'
    return t


def t_OF(t):
    r'\bof\b'
    return t


def t_DONE(t):
    r'\bdone\b'
    return t


def t_PROGRAM(t):
    r'\bprogram\b'
    return t


def t_DEFAULT(t):
    r'\bdefault\b'
    return t


def t_IDENTIFIER(t):
    r'\b[a-zA-Z_][0-9a-zA-Z_]{0,31}\b'
    return t


# bayad cast konim
def t_REALNUMBER(t):
    r'(-?[0-9]*\.[0-9]+)'
    return t


# ba . kaharabe
def t_NUMBER(t):
    r'-?[0-9]+'
    t.value = int(t.value)
    return t


def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'


# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


precedence = (
    ('right', 'OR'),
    ('right', 'AND'),
    ('right', 'UNOT'),
    ('nonassoc', 'LT', 'GT', 'EQ', 'NE', 'LE', 'GE'),
    ('right', 'PLUS', 'MINUS'),
    ('right', 'MUL', 'DIVIDE'),
    ('nonassoc', 'MOD'),
    ('right', 'UMINUS')
)


class Manager():
    def __init__(self):
        self.temp_int = 0
        self.temp_float = 0
        self.line = 0
        self.lines = []

    def next_line(self):
        self.line += 1
        return self.line

    def get_temp(self, type):
        if type == "int":
            return self.get_temp_int()
        if type == "float":
            return self.get_temp_float()

    def get_temp_int(self):
        self.temp_int += 1
        return f"temp_int{self.temp_int}"

    def get_temp_float(self):
        self.temp_float += 1
        return f"temp_float{self.temp_float}"

    def backpatch(self, labels, mquad):
        for label in labels:
            self.lines[label - 1] = (label, self.lines[label - 1][1].replace("LINE_NUMBER", str(mquad)))


manager = Manager()


def p_program(p):
    'program : PROGRAM IDENTIFIER declarations compound-statement'
    print(inspect.stack()[0][3])


def p_empty(p):
    'empty :'
    print(inspect.stack()[0][3])


def p_declarations(p):
    'declarations : VAR declaration-list'
    print(inspect.stack()[0][3])


def p_declarations2(p):
    'declarations : empty'
    print(inspect.stack()[0][3])


def p_declaration_list(p):
    'declaration-list : identifier-list COL type'
    print(inspect.stack()[0][3])


def p_declaration_list2(p):
    'declaration-list : declaration-list SEMICOL identifier-list COL type'
    print(inspect.stack()[0][3])


def p_identifier_list(p):
    'identifier-list : IDENTIFIER'
    print(inspect.stack()[0][3])


def p_identifier_list2(p):
    'identifier-list : identifier-list COMMA IDENTIFIER'
    print(inspect.stack()[0][3])


def p_type(p):
    'type : INT '
    print(inspect.stack()[0][3])


def p_type2(p):
    'type : REAL '
    print(inspect.stack()[0][3])


def p_compound_statement(p):
    'compound-statement : BEGIN statement-list END'
    print(inspect.stack()[0][3])


def p_statement_list(p):
    'statement-list : statement'
    print(inspect.stack()[0][3])


def p_statement_list2(p):
    'statement-list : statement-list SEMICOL statement'
    print(inspect.stack()[0][3])


def p_statement(p):
    'statement : IDENTIFIER ASSIGNMENT expression'
    print(inspect.stack()[0][3])


# def p_statement2(p):
#     'statement : IF expression THEN statement ELSE statement'
#     print(inspect.stack()[0][3])


# def p_statement3(p):
#     'statement : IF expression THEN statement'
#     print(inspect.stack()[0][3])

#####################################################
def p_m(t):
    'm : '
    # t[0] = nextinstr()

def p_n(t):
    'n : '
    # t[0] = nextinstr()
    # gen ("goto-")

def p_statement2(p):
    'statement : IF expression THEN m statement n ELSE m statement'
    # backpatch(p[2].tlist, p[4])
    # backpatch(p[2].flist, p[8])
    p[0].nlist = copy(p[5].nlist + p[9].nlist + p[6])
    print(inspect.stack()[0][3])

def p_statement3(p):
    'statement : IF expression THEN m statement'
    # backpatch(p[2].tlist, p[4])
    p[0].nlist = copy(p[2].flist + p[5].nlist)
    print(inspect.stack()[0][3])

def p_statement4(p):
    'statement : WHILE m expression DO m statement'
    # backpatch(p[6].nlist, p[2])
    # backpatch(p[3].tlist, p[5])
    p[0].nlist = copy(p[3].flist)
    # gen (f"goto {p[2]}")
    print(inspect.stack()[0][3])
##########################################################


# def p_statement4(p):
#     'statement : WHILE expression DO statement'
#     print(inspect.stack()[0][3])


def p_statement5(p):
    'statement : compound-statement'
    print(inspect.stack()[0][3])


def p_statement6(p):
    'statement : PRINT LPAREN expression RPAREN'
    print(inspect.stack()[0][3])


def p_statement7(p):
    'statement : SWITCH expression OF cases default-case DONE'
    print(inspect.stack()[0][3])


def p_default_case1(p):
    'default-case : DEFAULT statement SEMICOL'
    print(inspect.stack()[0][3])


def p_default_case2(p):
    'default-case : empty'
    print(inspect.stack()[0][3])


def p_cases1(p):
    'cases : constant-list COL statement SEMICOL cases'
    print(inspect.stack()[0][3])


def p_cases2(p):
    'cases : empty'
    print(inspect.stack()[0][3])


def p_constant_list1(p):
    'constant-list : constant'
    print(inspect.stack()[0][3])


def p_constant_list2(p):
    'constant-list : constant-list COMMA constant'
    print(inspect.stack()[0][3])


def p_constant_real(p):
    'constant : REALNUMBER'
    print(inspect.stack()[0][3])


def p_constant_int(p):
    'constant : NUMBER'
    print(inspect.stack()[0][3])


class Expression:
    def __init__(self, type="", value="", code=[], tlist=[], flist=[]):
        self.type = type
        self.value = value
        self.code = copy(code)
        self.tlist = copy(tlist)
        self.flist = copy(flist)

    def copy(self, exp):
        self.type = exp.type
        self.value = exp.value
        self.code = copy(exp.code)
        self.tlist = copy(exp.tlist)
        self.flist = copy(exp.flist)

    def gen(self):
        for line in self.code:
            manager.lines.append(line)
        # print(self.code)

    @staticmethod
    def exp_arithmetic(exp1: Expression, exp2: Expression, op: str):
        type_ = Expression.calc_type(exp1.type, exp2.type, check="arithmetic")
        temp = manager.get_temp(type_)
        code = [(manager.next_line(), f"{temp} = {exp1.value} {op} {exp2.value};")]
        return Expression(type_, temp, code)

    @staticmethod
    def exp_relop(exp1: Expression, exp2: Expression, op: str):
        type_ = "bool"  # CHECK
        tlist = [manager.next_line()]
        flist = [manager.next_line()]
        code = [(tlist[0], f"if ({exp1.value} {op} {exp2.value}) goto LINE_NUMBER;"), (flist[0], "goto LINE_NUMBER;")]
        return Expression(type=type_, code=code, tlist=tlist, flist=flist)

    @staticmethod
    def calc_type(type1, type2, check="arithmetic"):
        if check == "arithmetic":
            if (type1 == "float" and type2 == "float") or (type1 == "float" and type2 == "int") or (
                    type1 == "int" and type2 == "float"):
                return "float"
            if type1 == "int" and type2 == "int":
                return "int"

        if check == "relop":
            if type1 == "bool" and type2 == "bool":
                return "bool"

        # TODO: Error
        return "typeError"


def p_expression_int(p):
    'expression : NUMBER'
    p[0] = Expression("int", p[1])


def p_expression_real(p):
    'expression : REALNUMBER'
    p[0] = Expression("float", p[1])


def p_expression_id(p):
    'expression : IDENTIFIER'
    p[0] = Expression("id", p[1])  # type???


def p_expression_plus(p):
    'expression : expression PLUS expression'
    p[0] = Expression.exp_arithmetic(p[1], p[3], "+")
    p[0].gen()


def p_expression_minus(p):
    'expression : expression MINUS expression'
    p[0] = Expression.exp_arithmetic(p[1], p[3], "-")
    p[0].gen()


def p_expression_mul(p):
    'expression : expression MUL expression'
    p[0] = Expression.exp_arithmetic(p[1], p[3], "*")
    p[0].gen()


def p_expression_div(p):
    'expression : expression DIVIDE expression'
    p[0] = Expression.exp_arithmetic(p[1], p[3], "/")
    p[0].gen()


def p_expression_umin(p):
    'expression :  MINUS expression %prec UMINUS'
    type_ = p[2].type
    temp = manager.get_temp(type_)
    code = [(manager.next_line(), f"{temp} = -{p[2].value}")]
    p[0] = Expression(type=type_, value=temp, code=copy(code))
    p[0].gen()


def p_expression_mod(p):
    'expression : expression MOD expression'
    if p[1].type != "int" or p[3].type != "int":
        print("TypeError")
    p[0] = Expression.exp_arithmetic(p[1], p[3], "%")
    p[0].gen()


def p_expression_lt(p):
    'expression : expression LT expression'
    p[0] = Expression.exp_relop(p[1], p[3], "<")
    p[0].gen()


def p_expression_eq(p):
    'expression : expression EQ expression'
    p[0] = Expression.exp_relop(p[1], p[3], "==")
    p[0].gen()


def p_expression_gt(p):
    'expression : expression GT expression'
    p[0] = Expression.exp_relop(p[1], p[3], ">")
    p[0].gen()


def p_expression_ne(p):
    'expression : expression NE expression'
    p[0] = Expression.exp_relop(p[1], p[3], "!=")
    p[0].gen()


def p_expression_le(p):
    'expression : expression LE expression'
    p[0] = Expression.exp_relop(p[1], p[3], "<=")
    p[0].gen()


def p_expression_ge(p):
    'expression : expression GE expression'
    p[0] = Expression.exp_relop(p[1], p[3], ">=")
    p[0].gen()


# def p_expression_and(p):
#     'expression : expression AND expression'
#     print(inspect.stack()[0][3])


# def p_expression_or(p):
#     'expression : expression OR expression'
#     print(inspect.stack()[0][3])


#####################################################################################

def p_expression_and(p):
    'expression : expression AND m expression'
    # backpatch(p[1].tlist, p[3])
    p[0].tlist = copy(p[4].tlist)
    p[0].flist = copy(p[1].flist + p[4].flist)
    print(inspect.stack()[0][3])

def p_expression_or(p):
    'expression : expression OR m expression'
    # backpatch(p[1].flist, p[3])
    p[0].tlist = copy(p[1].tlist + p[4].tlist)
    p[0].flist = copy(p[4].flist)
    print(inspect.stack()[0][3])

#######################################################################################


def p_expression_unot(p):
    'expression : NOT expression %prec UNOT'
    type_ = "bool"  # CHECK
    tlist = copy(p[2].flist)
    flist = copy(p[2].tlist)
    p[0] = Expression(type=type_, tlist=tlist, flist=flist)
    print(p[0].tlist)


def p_expression_par(p):
    'expression : LPAREN expression RPAREN'
    p[0] = Expression()
    p[0].copy(p[2])


# Error rule for syntax errors
def p_error(p):
    print("Syntax error in input!")


# Build the lexer


lexer = lex.lex()
parser = yacc.yacc(start="expression")

# Test it out
data = '''
2
'''

# Give the lexer some input
lexer.input(data)

# Tokenize
while True:
    tok = lexer.token()
    if not tok:
        break  # No more input
    print(tok)

s = ''' 
not (5+9)/4 <> 8 + -(4)

 '''
print("input: ", s)

result = parser.parse(s)
manager.backpatch([6], 7)
for line in manager.lines:
    print(line)

