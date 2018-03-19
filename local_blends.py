import pandas as pd
from sklearn.preprocessing import minmax_scale
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df_9836 = pd.read_csv('sub_h_n_consolidated_filtered_inter.csv')
df_9830 = pd.read_csv('gru_attn_300p_300d_lc_submission.csv')
df_9782 = pd.read_csv('submission-lr-no-cw.csv')
df_svm = pd.read_csv('submission-lsvm-no-cw.csv')
df_han_cnn = pd.read_csv('han_class_cnn.csv')

for label in labels:
    df_9836[label] = minmax_scale(df_9836[label])
    df_9830[label] = minmax_scale(df_9830[label])
    df_9782[label] = minmax_scale(df_9782[label])
    df_svm[label] = minmax_scale(df_svm[label])
    df_han_cnn[label] = minmax_scale(df_han_cnn[label])
        
submission = pd.DataFrame()
submission['id'] = df_9836['id']
"""
submission[labels] = (df_9836[labels]*9 + \
                     df_9830[labels]*9 + \
                     df_9782[labels]*2) / 20



submission[labels] = (df_9836[labels]*8 + \
                     df_9830[labels]*7 + \
                     df_9782[labels]*5) / 20
"""
submission[labels] = (df_9836[labels]*6 +df_9830[labels]*6 + df_han_cnn[labels]*6) / 18
#for label in labels:
#    submission[label] = minmax_scale(submission[label])

submission.to_csv('tc_blend_weighted_lrhigh_hancnn.csv', index=False)