import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#ws = ["w_0.txt","dense_1_weights_dump.txt","dense_2_weights_dump.txt","dense_3_weights_dump.txt"]
#bs = ["dense_0_bias_dump.txt","dense_1_bias_dump.txt","dense_2_bias_dump.txt","dense_3_bias_dump.txt",]
ws = ["w_0.txt","w_1.txt","w_2.txt","w_3.txt","w_4_trim.txt"]
bs = ["b_0.txt","b_1.txt","b_2.txt","b_3.txt","b_4_trim.txt"]


def data_analysis(arr, save_img_name=None, plot=False):
    '''
    Array data analysis.
    '''
    df = pd.DataFrame(np.array(arr).reshape(-1, 1), columns=['val'])
    # \u7edf\u8ba1\u5143\u7d20\u503c\u7684\u4e2a\u6570
    pd.value_counts(df.values.ravel())
    df.val.hist(grid=True, bins=20, figsize=(8, 4))
    # print(pd.value_counts(df.values.ravel()))

    plt.ylabel('Apperances num, mean={:.2}, std={:.2}'.format(df["val"].value_counts().mean(), df["val"].value_counts().std()))
    plt.xlabel('Values, distri[{:.3},{:.3}]'.format(float(min(df['val'].values)), float(max(df['val'].values))))
    plt.title('Data distributions.')
    if save_img_name is not None:
        plt.savefig(save_img_name)
    if plot is True:
        plt.show(block=True)
    # Clear repeated graph.
    plt.clf()


if __name__ == "__main__":
    for f in ws:
        w = np.loadtxt(f,dtype=np.int)
        data_analysis(w,save_img_name=f+".png")
    
    
