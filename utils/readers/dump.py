import argparse

from .klp import KLP
from .ppin import PPIN


if __name__ == '__main__':
    def console():
        parser = argparse.ArgumentParser()
        parser.add_argument('--klp', default=None)
        parser.add_argument('--ppin', default=None)
        parser.add_argument('--cvoice', default=None)
        parser.add_argument('--proc', default=None, type=int)
        args = parser.parse_args()

        if args.klp is not None:
            klp = KLP(args.klp)
            klp.dump('./datasets/klp.pickle')

        if args.ppin is not None:
            ppin = PPIN(args.ppin)
            ppin.dump('./datasets/ppin.pickle')
        
        if args.cvoice is not None:
            # hugginface datasets
            from datasets import load_dataset, get_dataset_config_names

            DATASET = 'mozilla-foundation/common_voice_11_0'
            languages = get_dataset_config_names(DATASET)
            
            for lang in languages:
                load_dataset(DATASET, lang, cache_dir=args.cvoice, num_proc=args.proc)

    console()
