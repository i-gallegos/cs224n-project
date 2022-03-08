import sys
sys.path.append('../../')

import pandas as pd
import random
import evalRouge
random.seed(0)

dir = '..\\..\\access\\access\\preds\\post_summarized\\'

for dataset in ['tosdr']: # tldr, small_billsum
    out_file = dir+'preds_'+dataset+'_test_postsum'
    ref_file = '..\\..\\data\\reference\\reference_'+dataset+'_test.txt'
    result_file = 'rouge_'+dataset+'_test.csv'

    print(out_file)
    print(ref_file)
    print(result_file)

    df = pd.DataFrame(columns=['R-1', 'R-2', 'R-L'])

    preds = out_file
    true = ref_file

    rouge = evalRouge.eval(preds, true)
    df = pd.DataFrame.from_dict({'R-1':[rouge['rouge-1']['f']],
                                 'R-2':[rouge['rouge-2']['f']],
                                 'R-L':[rouge['rouge-l']['f']]})

    df.to_csv(result_file, index=False)
