from ucca import core, layer0, layer1
from ucca.layer1 import EdgeTags
from .format import FormatConverter

import sys
import re


class PtbConverter(FormatConverter):
    def __init__(self):
        self.node_by_id = {}
        self.terminals = []
        self.pending_nodes = []
        self.node_ids_with_children = set()
        self.sentence_id = 1
        self.passage_id = 3

    def _build_passage(self, stream):
        # p = core.Passage(self.sentence_id or self.passage_id)
        p = core.Passage(self.passage_id)
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)
        paragraph = 1

        next(self.parse(stream))

        # add normal nodes
        self.pending_nodes = list(reversed(self.pending_nodes))
        while self.pending_nodes:
            for i in reversed(range(len(self.pending_nodes))):
                parent_id, edge_tag, node_id = self.pending_nodes[i]
                parent = self.node_by_id.get(parent_id, -1)
                if parent != -1:
                    del self.pending_nodes[i]
                    implicit = node_id not in self.node_ids_with_children
                    node = l1.add_fnode(parent, edge_tag, implicit=implicit)
                    if edge_tag == EdgeTags.Punctuation:
                        node.tag = layer1.NodeTags.Punctuation
                    self.node_by_id[node_id] = node

        # add terminals
        for text, tag, edge_tag, parent_id in self.terminals:
            punctuation = (tag == layer0.NodeTags.Punct)
            terminal = l0.add_terminal(text=text, punct=punctuation, paragraph=paragraph)
            try:
                parent = self.node_by_id[parent_id]
            except KeyError as e:
                raise ValueError("Terminal ('%s') with bad parent (%s) in passage %s" % (text, parent_id, p.ID)) from e
            if parent is None:
                print("Terminal is a child of the root: '%s'" % text, file=sys.stderr)
                parent = l1.add_fnode(parent, edge_tag)
            if edge_tag != EdgeTags.Terminal:
                print("Terminal with incoming %s edge: '%s'" % (edge_tag, text), file=sys.stderr)
            parent.add(EdgeTags.Terminal, terminal)
        return p

    def from_format(self, stream, passage_id, return_original=False, **kwargs):
        self.passage_id = passage_id
        passage = self._build_passage(stream)
        passage.extra["format"] = kwargs.get("format") or "ptb"
        # yield (passage, self.lines_read, passage.ID) if return_original else passage
        yield passage
        self.node_by_id = None
        self.lines_read = []

    def parse(self, line_or_lines):
        def istok(t, i):
            return getattr(t, 'token_id', None) is i

        stack = []
        node_id = 0
        for tok in lex(line_or_lines):
            if tok.token_id is LPAREN_TOKEN:
                stack.append(tok)
            elif tok.token_id is STRING_TOKEN:
                stack.append(tok)
            # right parentheses, wrap up tree node
            else:
                # if terminal
                if (istok(stack[-1], STRING_TOKEN) and
                        istok(stack[-2], STRING_TOKEN) and
                        istok(stack[-3], LPAREN_TOKEN)):
                    w = Leaf(stack[-1].value, stack[-2].value, parent_id=node_id)
                    stack.pop()
                    stack.pop()
                    stack.pop()
                    stack.append(TExpr(w, node_id=node_id))
                    self.terminals.append((w.word, w.tag, w.edge_tag, node_id))
                # if regular node
                else:
                    tx = None
                    tail = None
                    peers = []
                    while not istok(stack[-1], LPAREN_TOKEN):
                        head = stack.pop()
                        # if it is the edge tag
                        if istok(head, STRING_TOKEN):
                            tx = TExpr(
                                Symbol(head.value), first_child=tail, next_sibling=None, node_id=node_id
                            )
                        # if it is a TExp object
                        else:
                            peers.append(head)
                            head.next_sibling = tail
                            tail = head
                    stack.pop()

                    if tx is None:
                        tx = TExpr(None, node_id=node_id)

                    tx.child_list = peers
                    for child in tx.children():
                        child.parent_node = tx
                        # if the child is a terminal
                        if isinstance(child.head, Leaf):
                            self.node_ids_with_children.add(child.node_id)
                            self.pending_nodes.append((child.parent_node.node_id, str(child.head.pos), child.node_id))
                            self.node_ids_with_children.add(child.parent_node.node_id)
                        # if the child is a regular node
                        else:
                            self.pending_nodes.append((child.parent_node.node_id, str(child.head), child.node_id))
                            self.node_ids_with_children.add(child.parent_node.node_id)
                            # node is root
                            if child.parent_node.node_id is None:
                                self.node_by_id[child.node_id] = None
                    if not stack:
                        self.node_by_id[node_id] = None
                        yield tx
                    else:
                        stack.append(tx)
                node_id += 1


#######
# Utils
#######

def gensym():
    return object()


##################
# Lexer
##################


LPAREN_TOKEN = gensym()
RPAREN_TOKEN = gensym()
STRING_TOKEN = gensym()

class Token(object):
    _token_ids = {LPAREN_TOKEN:"(", RPAREN_TOKEN:")", STRING_TOKEN:"STRING"}

    def __init__(self, token_id, value=None, lineno=None):
        self.token_id = token_id
        self.value = value
        self.lineno = lineno

    def __str__(self):
        return "Token:'{tok}'{ln}".format(
            tok=(self.value if self.value is not None else self._token_ids[self.token_id]),
            ln=(':{}'.format(self.lineno) if self.lineno is not None else '')
            )


_token_pat = re.compile(r'\(|\)|[^()\s]+')
def lex(line_or_lines):
    """
    Create a generator which returns tokens parsed from the input.
    The input can be either a single string or a sequence of strings.
    """

    if isinstance(line_or_lines, str):
        line_or_lines = [line_or_lines]

    counter = 0
    for n, line in enumerate(line_or_lines):
        line.strip()
        for m in _token_pat.finditer(line):
            if m.group() == '(':
                # if there is an extra '(' at the beginning of the sentence
                if counter == 1:
                    continue
                counter += 1
                yield Token(LPAREN_TOKEN)
            elif m.group() == ')':
                counter += 1
                yield Token(RPAREN_TOKEN)
            else:
                counter += 1
                yield Token(STRING_TOKEN, value=m.group())


##################
# Parser
##################


class Symbol:
    _pat = re.compile(r'(?P<label>^[^0-9=-]+)|(?:-(?P<tag>[^0-9=-]+))|(?:=(?P<parind>[0-9]+))|(?:-(?P<coind>[0-9]+))')

    def __init__(self, label):
        self.label = label
        self.tags = []
        self.coindex = None
        self.parindex = None
        for m in self._pat.finditer(label):
            if m.group('label'):
                self.label = m.group('label')
            elif m.group('tag'):
                self.tags.append(m.group('tag'))
            elif m.group('parind'):
                self.parindex = m.group('parind')
            elif m.group('coind'):
                self.coindex = m.group('coind')

    def simplify(self):
        self.tags = []
        self.coindex = None
        self.parindex = None

    def __str__(self):
        return '{}{}{}{}'.format(
            self.label,
            ''.join('-{}'.format(t) for t in self.tags),
            ('={}'.format(self.parindex) if self.parindex is not None else ''),
            ('-{}'.format(self.coindex) if self.coindex is not None else '')
        )


PUNC = [',', '.']
class Leaf:
    def __init__(self, word, pos, parent_id, edge_tag = "Terminal", node_id = None):
        self.word = word
        self.pos = pos
        self.parent_id = parent_id
        self.edge_tag = edge_tag
        self.tag = self.is_punc(word)
        self.node_id = node_id
        # self.is_punc = (self.pos == layer0.NodeTags.Punct)

    def __str__(self):
        return '({} {})'.format(self.pos, self.word)

    def is_punc(self, word):
        if word in PUNC:
            return "Punctuation"
        else:
            return "Word"

class TExpr:
    def __init__(self, head, first_child = None, next_sibling=None, node_id=None):
        self.head = head
        self.child_list = []
        self.parent_node = None
        self.node_id = node_id
        self.first_child = first_child
        self.next_sibling = next_sibling

    def symbol(self):
        if hasattr(self.head, 'label'):
            return self.head
        else:
            return None

    def parent(self):
        return self.parent_node

    def children(self):
        return self.child_list

    def leaf(self):
        if hasattr(self.head, 'pos'):
            return self.head
        else:
            return None

    def rule(self):
        if self.leaf():
            return '{} -> {}'.format(self.leaf().pos, self.leaf().word)
        else:
            return '{} -> {}'.format(self.symbol(), ' '.join(str(c.symbol() or c.leaf().pos) for c in self.children()))
