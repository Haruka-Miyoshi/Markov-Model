import numpy as np

"""マルコフモデル"""
class MarkovModel():
    """コンストラクタ"""
    def __init__(self, data:list, category:list):
        # サイコロの種類 遷移行列A[wi, wj]
        self._A = None
        # サイコロの目 遷移行列B[wi, vk]
        self._B = None
        # 確率P(s1)
        self._row = np.zeros(3)
        # サイコロの種類 頻度
        self._cat_hist = np.zeros((3,3))
        # サイコロの目 頻度
        self._dat_hist = np.zeros((3,2))
        # サイコロの種類 データ
        self._cat = category
        # サイコロの目 データ
        self._dat = data

    """出現回数を計算"""
    def calc_hist(self):
        # 出現回数を計算
        N = len(self._cat)

        for n in range(1, N):
            state = int(self._cat[n-1])
            next_state = int(self._cat[n])
            self._cat_hist[state, next_state] +=1

        for t in range(N):
            cat_ = int(self._cat[t])
            dat_ = int(self._dat[t]) - 1
            self._dat_hist[cat_, dat_] += 1
    
    """パラメータ推定"""
    def parameter_inference(self):
        # 出現回数を計算
        self.calc_hist()

        # 遷移行列Aの推定
        self._A = self._cat_hist
        self._A = self._A.T / np.sum(self._A, axis=1)
        self._A = self._A.T
        print("遷移行列A:" + "\n" + f"{self._A}" + "\n")

        # 遷移行列Bの推定
        self._B = self._dat_hist
        self._B = self._B.T / np.sum(self._B, axis=1)
        self._B = self._B.T
        print("遷移行列B:" + "\n" + f"{self._B}" + "\n")

        # 確率P(s1)の推定
        index = int(self._cat[0])
        self._row[index] += 1
        print("確率分布P(s1):" + "\n" + f"{self._row}")

"""実行文"""
if __name__ == '__main__':
    # サイコロの種類
    category = np.loadtxt("./data/categorys.txt")
    # サイコロの目
    data = np.loadtxt("./data/dataset.txt")
    # マルコフモデル
    mm = MarkovModel(data, category)
    # パラメータ推定 A, B, row
    mm.parameter_inference()