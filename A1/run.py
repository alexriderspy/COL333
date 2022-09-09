import json
import pickle
import argparse
import utils
from solvers import SentenceCorrector


def read_cli():
    parser = argparse.ArgumentParser(description='Run sentence corrector.')
    parser.add_argument(
        "-src",
        "--src_file",
        help="Source file containing input strings",
        required=False,
        type=str,
        default='./data/input.txt',
    )
    parser.add_argument(
        "-tar",
        "--tar_file",
        help="Target file where output strings are stored",
        required=False,
        type=str,
        default='./data/pred.txt',
    )
    parser.add_argument(
        "-cmat",
        "--confusion_matrix",
        help="Path to confusion matrix",
        required=False,
        type=str,
        default='./data/conf_matrix.json',
    )
    parser.add_argument(
        "-lm",
        "--lm_file",
        help="Path to language model",
        required=False,
        type=str,
        default='./data/lm_model.pkl',
    )
    parser.add_argument(
        "-tm",
        "--time_out",
        help="Timout value",
        required=False,
        type=float,
        default=-1,
    )
    args = vars(parser.parse_args())

    return args


def run_sentence_correction():
    args = read_cli()
    src_file = args['src_file']
    tar_file = args['tar_file']
    conf_file = args['confusion_matrix']
    lm_file = args['lm_file']
    time_out = args['time_out']

    with open(src_file, 'r') as fp:
        lines = [x.strip() for x in fp]

    with open(conf_file, 'r') as fp:
        conf_matrix = json.load(fp)

    with open(lm_file, 'rb') as fp:
        lm_model = pickle.load(fp)

    lm_model.unk_prob = 1e-20
    lm_model.set_mode('spell_check')

    preds = []
    for line in lines:
        sol = SentenceCorrector(lm_model, conf_matrix)
        try:
            utils.run_solver_with_timeout(sol, line, time_out)
        except TimeoutError:
            print(f'Solver timed out')
        
        pred = sol.best_state
        preds.append(pred)

    with open(tar_file, 'w') as fp:
        fp.writelines([f'{x}\n' for x in preds])


if __name__ == '__main__':
    run_sentence_correction()
