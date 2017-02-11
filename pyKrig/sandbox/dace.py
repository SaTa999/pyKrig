#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 02:15:37 2015

@author: Satoshi Takanashi
"""
from itertools import combinations
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf
import seaborn as sns
import os
try:
    import myutils
    _ENABLE_UTILS = True
except ImportError as e:
    print(e)
    print("compile myutils.pyx for faster evaluation")
    _ENABLE_UTILS = False

class Krig(object):
    """Kriging法により応答局面を求めるクラス"""
    _UTILS = _ENABLE_UTILS

    def __init__(self, table=None, btheta=None, EI=False, ANOVA=False):
        # table, input, bthetaを読み込む
        if table is None:
            try:
                self.table = np.loadtxt("table.csv", skiprows=1, delimiter=",")
            except IOError:
                self.table = np.loadtxt("table.txt", skiprows=1)
        else:
            self.table = table
        inputopt = np.loadtxt("inputopt.txt", delimiter="\n", dtype=bytes)
        self.nobj = int(inputopt[7].split()[0]) # 目的関数の数（現在は1しか対応していません）
        self.ndv = int(inputopt[8].split()[0]) # 設計変数の数
        if btheta is None:
            try:
                self.btheta = np.loadtxt("thetamax.csv", delimiter=",")
            except IOError:
                print("btheta.csv not found")
                self.btheta = 5. * np.ones(self.ndv)
        else:
            self.btheta = btheta
        self.dv_minmax = np.array([list(map(float, l.split()[:2])) for l in inputopt[9:9 + self.ndv]]) # 設計変数の最大・最小
        x = self.table[:, 1:1 + self.ndv]  # tableの変数の値をまとめた行列
        self.normx = self.normalizex(x) # tableの変数の値を正規化
        self.f = self.table[:, self.ndv+1]  # tableの目的関数の値をまとめた行列
        self.m = self.f.shape[0] # サンプル点数
        # myutilsを使用するか否か
        if Krig._UTILS:
            self.cloglikelihood = self._cloglikelihood2
            self.grad_cll = self._grad_cll2
        else:
            self.cloglikelihood = self._cloglikelihood
            self.grad_cll = self._grad_cll
            self.pairwise = np.square(self.normx[:, np.newaxis, :] - self.normx[np.newaxis, :, :])
        self.rmat = self.rmatrix()
        self.crmat = self.chol_rmatrix()
        self.mu = self.mu_()
        self.normf = self.f - self.mu
        self.irnf = self.invrmat_normf()
        if EI:
            self.irmat = self.invrmat()
            self.sigma = self.sigma_()
            self.iruv = self.invrmat_uvec()
            self.uviruv = np.sum(self.iruv) # 1^t * inv(R) * 1
            self.fmin = np.min(self.f)
            self.fmax = np.max(self.f)
            self.invsq2pi = 1. / math.sqrt(2. * math.pi)
            self.invsq2 = 1. / math.sqrt(2.)
        if ANOVA:
            self.farray = self.f_array()
            self.pfarray = self.prod_f_array()
            self.garray = self.g_array()
            self.pgarray = self.prod_g_array()
            self.pmu = self.predict_mu()
            self.pvar = self.predict_var()
            self.vars = {}
            self.calc_vars()

    def normalizex(self, x):
        """xを正規化する"""
        return (x - self.dv_minmax[:, 0]) / (self.dv_minmax[:, 1] - self.dv_minmax[:, 0])

    def denormalizex(self, x):
        """正規化されたxをもとに戻す"""
        return x * (self.dv_minmax[:, 1] - self.dv_minmax[:, 0]) +  self.dv_minmax[:, 0]

    def rmatrix(self):
        """i,j要素がCorr[xi,xj]となる行列"""
        if Krig._UTILS:
            return myutils.wpdist(self.normx, self.btheta)
        else:
            return np.exp(np.einsum("ijk,k", self.pairwise, -self.btheta, order="C"))

    def chol_rmatrix(self):
        """相関行列をコレスキー分解した行列"""
        if Krig._UTILS:
            crm = np.array(self.rmat, copy=True)
            myutils.chol_fact(crm)
            return (crm, False)
        else:
            return cho_factor(self.rmat, check_finite=False)

    def mu_(self):
        """Ln(μ,σ^2,θ)が最大となるときのμ"""
        crm = cho_factor(self.rmat, check_finite=False)
        unit_v = np.ones(self.m)
        rhs = np.vstack((self.f, unit_v)).T
        tmp = cho_solve(crm, rhs, check_finite=False)
        tmp2 = np.inner(unit_v, tmp.T)
        mu = tmp2[0] / tmp2[1]
        return mu

    def mu_2(self, crm):
        """任意のbthetaに対してのμ"""
        unit_v = np.ones(self.m)
        rhs = np.vstack((self.f, unit_v)).T
        tmp = cho_solve(crm, rhs, check_finite=False)
        mus = np.inner(unit_v, tmp.T)
        mu = mus[0] / mus[1]
        return mu

    def invrmat(self):
        """inv(R)"""
        return np.array(cho_solve(self.crmat, np.eye(self.m), check_finite=False), order="C")

    def invrmat_normf(self):
        """inv(R)*(y-1*mu)"""
        return cho_solve(self.crmat, self.normf, check_finite=False)

    def invrmat_uvec(self):
        """inv(R)*(1)"""
        uvec = np.ones(self.m)
        return cho_solve(self.crmat, uvec, check_finite=False)

    def sigma_(self):
        """Ln(μ,σ^2,θ)が最大となるときのσ^2"""
        sigma = np.inner(self.normf, self.irnf)
        sigma /= self.m
        return sigma

    def _cloglikelihood(self, btheta):
        """KrigingモデルのCompressed Loglikelihood（Jones[2001]）を計算(numpy+scipyを使用)
        L = -n / 2 * ln(σ ^ 2) - 1 / 2 * ln(det(R))"""
        rm = np.exp(np.einsum("ijk,k", self.pairwise, -btheta), order="C")
        crm = cho_factor(rm, check_finite=False)
        mu = self.mu_2(crm)
        normf = self.f - mu
        sigma = np.inner(normf, cho_solve(crm, normf, check_finite=False))
        sigma /= self.m
        ldrm = np.sum(np.log(np.diag(crm[0])))
        cllh = - 0.5 * (self.m * np.log(sigma)) - ldrm
        return cllh

    def _cloglikelihood2(self, btheta):
        """KrigingモデルのCompressed Loglikelihood（Jones[2001]）を計算(numpy+scipy+cython+LAPACKを使用)"""
        crm = myutils.wpdist(self.normx, btheta)
        myutils.chol_fact(crm)
        # mu
        uv = np.ones(crm.shape[0])
        fuv = np.array((self.f, uv), order="C")
        myutils.chol_solve(crm, fuv)
        mus = np.inner(uv, fuv)
        mu = mus[0] / mus[1]
        # sigma
        normf = self.f - mu
        tmp = fuv[0] - fuv[1] * mu
        sigma = np.inner(normf, tmp)
        sigma /= self.m
        # log(det(R))
        ldrm = np.sum(np.log(np.diag(crm)))
        cllh = - 0.5 * (self.m * np.log(sigma)) - ldrm
        return cllh

    def _grad_cll(self, btheta):
        """Compressed Loglikelihoodの勾配をReverse algorithmic differentationを用いて計算(Toal 2009)
        (numpy+scipyを使用)"""
        rm = np.exp(np.einsum("ijk,k", self.pairwise, -btheta, order="C"))
        crm = cho_factor(rm, check_finite=False)
        ir = cho_solve(crm, np.eye(self.m), check_finite=False)
        mu = self.mu_2(crm)
        normf = self.f - mu
        tmp = np.dot(normf, ir)
        sigma = np.inner(tmp, normf)
        sigma /= self.m
        arm = (tmp[:,np.newaxis] * tmp) / (2 * sigma) - 0.5 * ir
        gll = - np.einsum("ijk, ij, ij", self.pairwise, rm, arm)
        return gll

    def _grad_cll2(self, btheta):
        """Compressed Loglikelihoodの勾配をReverse algorithmic differentationを用いて計算(Toal 2009)
        (numpy+scipy+cython+LAPACKを使用)"""
        rm = myutils.wpdist(self.normx, btheta)
        crm = np.array(rm, copy=True)
        myutils.chol_fact(crm)
        # mu
        uv = np.ones(crm.shape[0])
        fuv = np.array((self.f, uv), order="C")
        myutils.chol_solve(crm, fuv)
        mus = np.inner(uv, fuv)
        mu = mus[0] / mus[1]
        # sigma
        myutils.chol_inv(crm)
        ir = np.asarray(crm)
        normf = self.f - mu
        tmp = myutils.symdot(ir, normf)
        sigma = np.inner(tmp, normf)
        sigma /= self.m
        gll = myutils.gll(tmp, self.normx, rm, ir, sigma)
        return gll

    def corr_vec(self, xi):
        """i番目の要素がCorr[e(x),e(xi)]となるm次元ベクトル
        xi: 正規化した座標"""
        return  np.exp(np.einsum("ij, j", (self.normx - xi) ** 2, -self.btheta))

    def predict(self, x):
        """Kriging法によって求めた応答局面上の点xにおける値"""
        return self.mu + np.inner(self.corr_vec(x), self.irnf)

    def mse(self, x):
        """点xiにおけるmean square error"""
        r = self.corr_vec(x)
        mse = (1. - np.inner(self.iruv, r)) ** 2
        mse /= self.uviruv
        mse += 1. - np.inner(r, np.dot(self.irmat, r))
        mse *= self.sigma
        return mse

    def rmse(self, x):
        """点xiにおけるroot mean square error、丸め誤差によりMSEが負になった場合は0.を返す"""
        mse = self.mse(x)
        rmse = math.sqrt(mse) if mse > 0. else 0.
        return rmse

    def pdf(self, x):
        """正規分布のprobability distribution function"""
        return math.exp(-0.5 * (x ** 2)) * self.invsq2pi

    def cdf(self, x):
        """正規分布のcumlative distribution function"""
        return 0.5 * (math.erf(x * self.invsq2) + 1.)

    def expected_improvement(self, x, minimize=True):
        """点xにおけるEI値を計算
        minimize: Trueなら最小化、Falseなら最大化
        rmseが0のときは0を返す"""
        if minimize:
            tmp = self.fmin - self.predict(x)
        else:
            tmp = self.predict(x) - self.fmax
        rmse = self.rmse(x)
        if rmse is 0.:
            return 0.
        else:
            tmp2 = tmp / rmse
            ei = tmp * self.cdf(tmp2) + rmse * self.pdf(tmp2)
            ei = ei if ei > 0. else 0.
            return ei

    def int_gauss(self, a, b):
        """gauss関数exp(-a*(x-b)^2)を区間[0,1]で積分した結果を返す（誤差関数を利用して計算）"""
        sa = np.sqrt(a)
        return 0.5 * np.sqrt(np.pi) / sa * (erf(sa * b) - erf(sa * (b - 1.)))

    def f_array(self):
        """i,j番目の要素がint_gauss(btheta, xi_j)となる(m, n_dv)次元配列"""
        return self.int_gauss(self.btheta, self.normx)

    def prod_f_array(self):
        """i番目の要素がΠ_j(_ij)となるベクトル"""
        return np.prod(self.farray, axis=-1)

    def int_square_gauss(self, a, b, c):
        """gauss関数exp(-a*((x-b)^2+(x-c)^2)を区間[0,1]で積分した結果を返す（誤差関数を利用して計算）"""
        sa = np.sqrt(a  * 0.5)
        return np.sqrt(np.pi) * 0.25 / sa * np.exp(-0.5*a*((b-c)**2)) * (erf(sa * (b + c)) - erf(sa * (b + c - 2.)))

    def g_array(self):
        """i,j,k番目の要素がint_square_gauss(btheta, xi_j, xi_k)]となる(m, m, n_dv)次元配列"""
        return self.int_square_gauss(self.btheta, self.normx, self.normx[:,np.newaxis,:])

    def prod_g_array(self):
        """i,j番目の要素がΠ_k(G_ijk)となる配列"""
        return np.prod(self.garray, axis=-1)

    def predict_mu(self):
        """Kriging法で求めた予測値から設計空間の平均値（mu^hat）を計算"""
        return self.mu + np.inner(self.pfarray, self.irnf)

    def predict_var(self):
        """Kriging法で求めた予測値から設計空間の分散（sigma^2^hat）を求める"""
        var = - (self.mu  - self.pmu) ** 2
        var += np.einsum("i,j,ij", self.irnf, self.irnf, self.pgarray)
        return var

    def predict_var_maineffect(self, dvk):
        """Kriging法で求めた予測値から設計変数dvkの主効果の分散を求める"""
        dvlabel = "dv{}".format(dvk+1)
        if dvlabel not in self.vars:
            var = - (self.mu  - self.pmu) ** 2
            d = self.pfarray / self.farray[:, dvk]
            var += np.einsum("i,j,i,j,ij", self.irnf, self.irnf, d, d, self.garray[:,:,dvk])
            self.vars[dvlabel] = var
        return self.vars[dvlabel]

    def predict_var_twowayinteraction(self, dvk, dvl):
        """Kriging法で求めた予測値から設計変数dvk,dvlの交互作用の分散を求める"""
        dvlabel = "dv{}-{}".format(dvk+1,dvl+1)
        if dvlabel not in self.vars:
            var = - (self.mu  - self.pmu) ** 2 - self.predict_var_maineffect(dvk) - self.predict_var_maineffect(dvl)
            d = self.pfarray / (self.farray[:, dvk] * self.farray[:, dvl])
            var += np.einsum("i,j,i,j,ij,ij", self.irnf, self.irnf, d, d, self.garray[:, :, dvk], self.garray[:, :, dvl])
            self.vars[dvlabel] = var
        return self.vars[dvlabel]

    def calc_vars(self):
        for i, j in combinations(list(range(self.ndv)), 2):
            self.predict_var_twowayinteraction(i, j)

    def sorted_vars(self, filter=True, tol=5e-2, include_higher_degree=True):
        """小さい順にソートした分散量とそのラベルを返す
        filter: tol以下の割合の分散量をまとめてothersに
        include_higher_degree: 3つ以上の変数の交互作用による分散を含める"""
        vars = list(self.vars.values())
        labels = list(self.vars.keys())
        if include_higher_degree:
            vars_tot = self.pvar
        else:
            vars_tot = sum(vars)
        vars = [v / vars_tot for v in vars]
        vars, labels = list(zip(*sorted(zip(vars, labels))))
        vars, labels = list(vars), list(labels)
        if filter:
            vars, labels = list(zip(*((v,l)  for (v,l) in zip(vars,labels) if v>=tol)))
            vars, labels = list(vars), list(labels)
            vars.insert(0, 1.-sum(vars))
            labels.insert(0, "others")
        else:
            if include_higher_degree:
                vars.insert(0, 1.-sum(vars))
                labels.insert(0, "higher_degree")
        return vars, labels

    def maineffect(self, dvj):
        """Kriging法で求めた予測値からdvj番目の設計変数の主効果の変動量を求める関数を返す"""
        mu = self.mu - self.pmu
        c = self.irnf
        d = self.pfarray / self.farray[:,dvj]
        tj = self.btheta[dvj]
        xj = self.normx[:,dvj]
        def _mefunc(x):
            """xはスカラでもベクトルでも"""
            tmp = np.exp(-tj * np.square(x - xj))
            return mu + np.inner(c, d * tmp)
        return _mefunc

    def twowayinteraction(self, dvj, dvk):
        """Kriging法で求めた予測値からdvj, dvk番目の設計変数の交互作用の変動量を求める関数を返す"""
        mu = self.mu - self.pmu
        c = self.irnf
        d = self.pfarray / (self.farray[:,dvj] * self.farray[:,dvk])
        tj = self.btheta[dvj]
        tk = self.btheta[dvk]
        xj = self.normx[:,dvj]
        xk = self.normx[:,dvk]
        def _twifunc(x1,x2):
            """x1,x2はスカラでもベクトルでも"""
            f1 = self.maineffect(dvj)
            f2 = self.maineffect(dvk)
            tmp = np.exp(- tj * np.square(x1 - xj) - tk * np.square(x2 - xk))
            return mu + np.inner(c, d * tmp) - f1(x1) - f2(x2)
        return _twifunc


def main():
    deg = 100 # グラフの各軸の刻み数

    # グラフの設定
    sns.set_context("paper", font_scale=1.8)
    sns.set_style("whitegrid", rc={"legend.frameon": True })
    sns.set_palette("Set1", n_colors=9, desat=0.8)

    krige = Krig(EI=False, ANOVA=True)

    # 分散の割合の円グラフ
    print("calc_vars")
    vars, labels = krige.sorted_vars()
    vars.reverse(), labels.reverse()
    np.savetxt("vars.csv", np.array(vars)[np.newaxis,:], header=",".join(labels), delimiter=",", comments="")
    plt.figure(figsize=(8,8))
    plt.pie(vars,labels=labels, startangle=90, autopct="%1.f%%", counterclock=False)
    plt.savefig("vars")
    plt.close("all")

    x1 = np.linspace(0,1,deg)
    X1, X2 = np.meshgrid(x1,x1)

    # 主効果のグラフ
    for i in range(krige.ndv):
        label = "dv{}".format(i+1)
        if label in labels:
            print("calc maineffect ", label)
            f = krige.maineffect(i)
            tmp = f(x1[:,np.newaxis])
            plt.plot(x1, tmp, label=label)
    plt.xticks(np.arange(0, 1.2, 0.2))
    # plt.grid()
    plt.legend(loc="best")
    # plt.show()
    plt.tight_layout(pad=0.5)
    plt.savefig("main_effect")
    plt.close("all")

    # 交互作用のグラフ
    for i,j in  combinations(list(range(krige.ndv)), 2):
        label = "dv{}-{}".format(i+1, j+1)
        if label in labels:
            print("calc twowayinteraction ", label)
            plt.figure(figsize=(6.4, 4.8))
            g = krige.twowayinteraction(i,j)
            tmp = g(X1[:,:,np.newaxis],X2[:,:,np.newaxis])
            mumax = max(np.max(tmp), abs(np.min(tmp)))
            v = np.linspace(-mumax, mumax, 11)
            plt.contourf(X1, X2, tmp, v, cmap="jet")
            plt.xticks(np.arange(0, 1.2, 0.2))
            plt.yticks(np.arange(0, 1.2, 0.2))
            plt.xlabel("dv{}".format(i + 1))
            plt.ylabel("dv{}".format(j + 1))
            plt.colorbar(ticks=v)
            # plt.show()
            plt.axis("scaled")
            plt.tight_layout(pad=0.1)
            plt.savefig("twoway_interaction_{}".format(label))
            plt.close("all")


if __name__ == '__main__':
    main()