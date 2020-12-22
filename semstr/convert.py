#!/usr/bin/env python3

import sys

import configargparse
import csv
import os
import re
from glob import glob
from tqdm import tqdm
from ucca import ioutil, layer1
from ucca.convert import from_text, to_text, from_json, to_json
from ucca.normalization import normalize

from semstr.cfgutil import add_verbose_arg, add_boolean_option
from semstr.validation import validate, print_errors

description = """Parses files in the specified format, and writes as the specified format.
Each passage is written to the file: <outdir>/<prefix><passage_id>.<extension> """


def from_conll(lines, passage_id, return_original=False, dep=False, preprocess=True, **kwargs):
    """Converts from parsed text in CoNLL format to a Passage object.

    :param lines: iterable of lines in CoNLL format, describing a single passage.
    :param passage_id: ID to set for passage
    :param return_original: return triple of (UCCA passage, CoNLL string, sentence ID)
    :param dep: return dependency graph rather than converted UCCA passage
    :param preprocess: preprocess the dependency graph before converting to UCCA (or returning it)?

    :return generator of Passage objects
    """
    from semstr.conversion.conll import ConllConverter
    return ConllConverter().from_format(lines, passage_id=passage_id, return_original=return_original, dep=dep,
                                        preprocess=preprocess, format=kwargs.get("format"))


def to_conll(passage, test=False, tree=False, preprocess=True, **kwargs):
    """ Convert from a Passage object to a string in CoNLL-X format (conll)

    :param passage: the Passage object to convert
    :param test: whether to omit the head and deprel columns. Defaults to False
    :param tree: whether to omit rows for non-primary parents. Defaults to False
    :param preprocess: preprocess the converted dependency graph before returning it?

    :return list of lines representing the dependencies in the passage
    """
    from semstr.conversion.conll import ConllConverter
    return ConllConverter(tree=tree).to_format(passage, test, preprocess=preprocess, format=kwargs.get("format"))


def from_export(lines, passage_id=None, return_original=False, **kwargs):
    """Converts from parsed text in NeGra export format to a Passage object.

    :param lines: iterable of lines in NeGra export format, describing a single passage.
    :param passage_id: ID to set for passage, overriding the ID from the file
    :param return_original: return triple of (UCCA passage, Export string, sentence ID)

    :return generator of Passage objects
    """
    from semstr.conversion.export import ExportConverter
    return ExportConverter().from_format(lines, passage_id=passage_id, return_original=return_original,
                                         format=kwargs.get("format"))


def to_export(passage, test=False, tree=False, **kwargs):
    """ Convert from a Passage object to a string in NeGra export format (export)

    :param passage: the Passage object to convert
    :param test: whether to omit the edge and parent columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to False

    :return list of lines representing a (discontinuous) tree structure constructed from the passage
    """
    from semstr.conversion.export import ExportConverter
    return ExportConverter().to_format(passage, test=test, tree=tree, format=kwargs.get("format"))


def from_amr(lines, passage_id=None, return_original=False, save_original=True, wikification=False, placeholders=True,
             **kwargs):
    """Converts from parsed text in AMR PENMAN format to a Passage object.

    :param lines: iterable of lines in AMR PENMAN format, describing a single passage.
    :param passage_id: ID to set for passage, overriding the ID from the file
    :param save_original: whether to save original AMR text in passage.extra
    :param return_original: return triple of (UCCA passage, AMR string, AMR ID)
    :param wikification: whether to use wikification for replacing node labels with placeholders based on tokens
    :param placeholders: introduce placeholders into node labels when they include the terminal's text?

    :return generator of Passage objects
    """
    from semstr.conversion.amr import AmrConverter
    return AmrConverter().from_format(lines, passage_id=passage_id, return_original=return_original,
                                      save_original=save_original, wikification=wikification, placeholders=placeholders,
                                      format=kwargs.get("format"))


def to_amr(passage, metadata=True, wikification=True, use_original=True, verbose=False, default_label=None,
           **kwargs):
    """ Convert from a Passage object to a string in AMR PENMAN format (export)

    :param passage: the Passage object to convert
    :param metadata: whether to print ::id and ::tok lines
    :param wikification: whether to wikify named concepts, adding a :wiki triple
    :param use_original: whether to use original AMR text from passage.extra
    :param verbose: whether to print extra information
    :param default_label: label to use in case node has no label attribute

    :return list of lines representing an AMR in PENMAN format, constructed from the passage
    """
    from semstr.conversion.amr import AmrConverter
    return AmrConverter().to_format(passage, metadata, wikification, verbose, use_original=use_original,
                                    default_label=default_label, format=kwargs.get("format"))


def from_conllu(lines, passage_id=None, return_original=False, annotate=False, terminals_only=False, dep=False,
                enhanced=True, preprocess=True, **kwargs):
    """Converts from parsed text in Universal Dependencies format to a Passage object.

    :param lines: iterable of lines in Universal Dependencies format, describing a single passage.
    :param passage_id: ID to set for passage
    :param return_original: return triple of (UCCA passage, Universal Dependencies string, sentence ID)
    :param annotate: whether to save dependency annotations in "extra" dict of layer 0
    :param terminals_only: create only terminals (with any annotation if specified), no non-terminals
    :param dep: return dependency graph rather than converted UCCA passage
    :param enhanced: whether to include enhanced edges
    :param preprocess: preprocess the dependency graph before converting to UCCA (or returning it)?

    :return generator of Passage objects
    """
    from semstr.conversion.conllu import ConlluConverter
    return ConlluConverter(enhanced=enhanced).from_format(lines, passage_id=passage_id, return_original=return_original,
                                                          annotate=annotate, terminals_only=terminals_only, dep=dep,
                                                          preprocess=preprocess, format=kwargs.get("format"))


def to_conllu(passage, test=False, enhanced=True, preprocess=True, **kwargs):
    """ Convert from a Passage object to a string in Universal Dependencies format (conllu)

    :param passage: the Passage object to convert
    :param test: whether to omit the head and deprel columns. Defaults to False
    :param enhanced: whether to include enhanced edges
    :param preprocess: preprocess the converted dependency graph before returning it?

    :return list of lines representing the semantic dependencies in the passage
    """
    from semstr.conversion.conllu import ConlluConverter
    return ConlluConverter(enhanced=enhanced).to_format(passage, test=test, preprocess=preprocess,
                                                        format=kwargs.get("format"))


def from_sdp(lines, passage_id, mark_aux=False, return_original=False, dep=False, preprocess=True, **kwargs):
    """Converts from parsed text in SemEval 2015 SDP format to a Passage object.

    :param lines: iterable of lines in SDP format, describing a single passage.
    :param passage_id: ID to set for passage
    :param mark_aux: add a preceding # for labels of auxiliary edges added
    :param return_original: return triple of (UCCA passage, SDP string, sentence ID)
    :param dep: return dependency graph rather than converted UCCA passage
    :param preprocess: preprocess the dependency graph before converting to UCCA (or returning it)?

    :return generator of Passage objects
    """
    from semstr.conversion.sdp import SdpConverter
    return SdpConverter(mark_aux=mark_aux).from_format(lines, passage_id=passage_id, return_original=return_original,
                                                       dep=dep, preprocess=preprocess, format=kwargs.get("format"))


def to_sdp(passage, test=False, tree=False, mark_aux=False, preprocess=True, **kwargs):
    """ Convert from a Passage object to a string in SemEval 2015 SDP format (sdp)

    :param passage: the Passage object to convert
    :param test: whether to omit the top, head, frame, etc. columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to False
    :param mark_aux: omit edges with labels with a preceding #
    :param preprocess: preprocess the converted dependency graph before returning it?

    :return list of lines representing the semantic dependencies in the passage
    """
    from semstr.conversion.sdp import SdpConverter
    return SdpConverter(mark_aux=mark_aux, tree=tree).to_format(passage, test=test, preprocess=preprocess,
                                                                format=kwargs.get("format"))


def from_ptb(lines, passage_id=None, return_original=False, **kwargs):
#def from_ptb(passage, passage_id=None, return_original=False, **kwargs):
    """Converts from parsed text in Penn Treebank mrg format to a Passage object.

    :param lines: iterable of lines in Penn Treebank mrg format, describing a single passage.
    :param passage_id: ID to set for passage, overriding the ID from the file
    :param return_original: return triple of (UCCA passage, Mrg string, sentence ID)

    :return generator of Passage objects
    """
    from semstr.conversion.ptb import PtbConverter
    return PtbConverter().from_format(lines, passage_id=passage_id, return_original=return_original,
                                      format=kwargs.get("format"))


def to_ptb(passage, test=False, tree=False, **kwargs):
    """ Convert from a Passage object to a string in Penn TreeBank mrg format (export)

    :param passage: the Passage object to convert
    :param test: whether to omit the edge and parent columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to False

    :return list of lines representing a (discontinuous) tree structure constructed from the passage
    """
    from semstr.conversion.ptb import PtbConverter
    return PtbConverter().to_format(passage, test=test, tree=tree, format=kwargs.get("format"))


CONVERTERS = {
    None: (None, None),
    "json": (from_json, to_json),
    "conll": (from_conll, to_conll),
    "conllu": (from_conllu, to_conllu),
    "sdp": (from_sdp, to_sdp),
    "export": (from_export, to_export),
    "amr": (from_amr, to_amr),
    "txt": (from_text, to_text),
    "ptb": (from_ptb, to_ptb),
    "mrg": (from_ptb, to_ptb),
}
FROM_FORMAT = {f: c[0] for f, c in CONVERTERS.items() if c[0] is not None}
TO_FORMAT = {f: c[1] for f, c in CONVERTERS.items() if c[1] is not None}

UCCA_EXT = (".xml", ".pickle")


def iter_files(patterns):
    for pattern in patterns:
        filenames = sorted(glob(pattern))
        if not filenames:
            raise IOError("Not found: " + pattern)
        yield from filenames


def iter_passages(patterns, desc=None, input_format=None, prefix="", label_map=None, output_format=None, **kwargs):
    t = tqdm(list(iter_files(patterns)), unit="file", desc=desc)
    for filename in t:
        t.set_postfix(file=os.path.basename(filename))
        if not os.path.isfile(filename):
            raise IOError("Not a file: %s" % filename)
        no_ext, ext = os.path.splitext(filename)
        if ext in UCCA_EXT:  # UCCA input
            yield ioutil.file2passage(filename)
        else:
            basename = os.path.basename(no_ext)
            try:
                passage_id = re.search(r"\d+(\.\d+)*", basename).group(0)
            except AttributeError:
                passage_id = basename
            converter = FROM_FORMAT.get(input_format or ext.lstrip("."), (from_text,))
            with open(filename, encoding="utf-8") as f:
                yield from converter(f, prefix + passage_id, format=output_format if label_map else None, **kwargs)


def map_labels(passage, label_map_file):
    if label_map_file:
        with open(label_map_file, encoding="utf-8") as f:
            label_map = dict(csv.reader(f))
        for node in passage.layer(layer1.LAYER_ID).all:
            for edge in list(node):
                mapped = label_map.get(edge.tag) or label_map.get(edge.tag.partition(":")[0])
                if mapped is None:
                    if edge.attrib.get("remote"):
                        node.remove(edge)
                else:
                    edge.tag = mapped
        try:
            del passage.extra["format"]  # Remove original format as it no longer applies, after labels were replaced
        except KeyError:
            pass


def write_passage(passage, out_dir=".", output_format=None, binary=False, verbose=False, label_map=False, split=False,
                  join=None, **kwargs):
    ext = {None: UCCA_EXT[binary], "amr": ".txt"}.get(output_format) or "." + output_format
    if join and join.endswith(ext):
        ext = ""
    outfile = os.path.join(out_dir, (join or passage.ID) + ext)
    if verbose:
        with ioutil.external_write_mode():
            print("Writing '%s'..." % outfile, file=sys.stderr)
    if output_format is None:  # UCCA output
        ioutil.passage2file(passage, outfile, binary=binary)
    else:
        converter = TO_FORMAT[output_format]
        with open(outfile, "a" if join else "w", encoding="utf-8") as f:
            for line in converter(passage, format=output_format if label_map else None, sentences=split, **kwargs):
                print(line, file=f)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    kwargs = vars(args)
    for passage in iter_passages(args.filenames, desc="Converting", **kwargs):
        map_labels(passage, args.label_map)
        if args.normalize and args.output_format != "txt":
            normalize(passage, extra=args.extra_normalization)
        if args.lang:
            passage.attrib["lang"] = args.lang
        write_passage(passage, **kwargs)
        if args.validate:
            try:
                errors = list(validate(passage, **kwargs))
            except ValueError:
                continue
            if errors:
                print_errors(errors, passage.ID)
                sys.exit(1)


def add_convert_args(p):
    add_boolean_option(p, "test", "omit prediction columns (head and deprel for conll; top, pred, frame, etc. for sdp)",
                       short="t")
    add_boolean_option(p, "tree", "remove multiple parents to get a tree", short="T")
    add_boolean_option(p, "split", "split each sentence to its own passage", short="s")
    add_boolean_option(p, "mark-aux", "mark auxiliary edges introduced/omit edges", short="m")
    p.add_argument("--label-map", help="CSV file specifying mapping of input edge labels to output edge labels")


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=description)
    argparser.add_argument("filenames", nargs="+", help="file names to convert")
    argparser.add_argument("-i", "--input-format", choices=CONVERTERS, help="input file format (detected by extension)")
    argparser.add_argument("-f", "--output-format", choices=CONVERTERS, help="output file format (default: UCCA)")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory")
    argparser.add_argument("-j", "--join", help="concatenate all output files to a file with this name")
    argparser.add_argument("-p", "--prefix", default="", help="output passage ID prefix")
    add_boolean_option(argparser, "binary", "write in binary format (.%s)" % UCCA_EXT[1], short="b")
    add_boolean_option(argparser, "annotate", "store dependency annotations in 'extra' dict", short="a")
    add_boolean_option(argparser, "validate", "validate every passage after conversion", short="V")
    add_boolean_option(argparser, "ucca-validation", "apply UCCA-specific validations", short="u")
    add_boolean_option(argparser, "enhanced", "read enhanced dependencies", default=True)
    add_boolean_option(argparser, "wikification", "AMR wikification", default=True)
    argparser.add_argument("--default-label", help="use this for missing AMR labels, otherwise raise exception")
    add_boolean_option(argparser, "normalize", "normalize passage", default=True)
    add_boolean_option(argparser, "extra-normalization", "more normalization rules")
    argparser.add_argument("-l", "--lang", help="small two-letter language code to set in output passage metadata")
    add_convert_args(argparser)
    add_verbose_arg(argparser, help="detailed output")
    main(argparser.parse_args())
    sys.exit(0)
