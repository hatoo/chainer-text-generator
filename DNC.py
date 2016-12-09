"""
Original code is https://github.com/yos1up/DNC .
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda, optimizers, Chain, Variable


def DNC_input_len(X, W, R):
    return W * R + X


def DNC_output_len(Y, W, R):
    return Y + W * R + 3 * W + 5 * R + 3


def C(M, k, beta):
    # "_" means length of mini-batch.
    # M : _ * N * W -- this type of comments indicate a shape of variable.
    # k : _ * W
    # beta : _ * 1
    # -> _ * N
    denominator = F.sqrt(
        F.scale(F.sum(M * M, 2), F.batch_l2_norm_squared(k), 0) + 1e-5)
    D = F.scale(F.reshape(F.batch_matmul(M, k),
                          (M.shape[0], M.shape[1])), 1 / denominator, 0)
    return F.softmax(D)


def Cr(M, k, beta):
    # M : _ * N * W
    # k : _ * R * W
    # beta : _ * R
    # -> _ * R * N
    [batch_size, R, N] = k.shape
    M_R = F.broadcast_to(F.reshape(M, (M.shape[0], 1, M.shape[1], M.shape[
                         2])), (batch_size, R, M.shape[1], M.shape[2]))
    # calculate denominator of D
    M_R_sum = F.sum(M_R * M_R, 3)
    k_sum = F.sum(k * k, 2)
    k_sum = F.reshape(k_sum, (batch_size, R, 1))
    denominator = F.sqrt(F.scale(M_R_sum, k_sum, 0) + 1e-5)
    cross = F.sum(
        M_R * F.broadcast_to(F.reshape(k, (M.shape[0], R, 1, N)), M_R.shape), 3)
    D = cross / denominator

    # calculate softmax
    D_exp = F.exp(D)
    D_denominator = F.sum(D_exp, 2)

    return F.scale(D_exp, 1 / F.reshape(D_denominator, (batch_size, R, 1)), 0)


def u2a(u):
    # u : _ * N
    # -> _ * N
    # TODO: Vectorize
    # TODO: Use GPU
    xp = cuda.get_array_module(u.data)
    [batch_size, N] = u.data.shape
    u_cpu = cuda.to_cpu(u.data)
    a = np.zeros(u.shape, dtype=np.float32)
    cumprod = np.ones(batch_size, dtype=np.float32)
    phi = np.argsort(u_cpu)
    for i in range(N):
        a[np.arange(batch_size), phi[:, i]] = cumprod * \
            (1.0 - u_cpu[np.arange(batch_size), phi[:, i]])
        cumprod *= u_cpu[np.arange(batch_size), phi[:, i]]

    # Discontinuities when calculating the gradient is OK.
    return Variable(xp.array(a), volatile='auto')


class DeepLSTM(Chain):  # too simple?

    def __init__(self, d_in, d_out):
        super(DeepLSTM, self).__init__(
            l1=L.LSTM(d_in, d_out),
            l2=L.Linear(d_out, d_out),)

    def __call__(self, x):
        self.x = x
        self.y = self.l2(self.l1(self.x))
        return self.y

    def reset_state(self):
        self.l1.reset_state()


class DNC(Chain):

    def __init__(self, X, Y, N, W, R, controller):
        self.X = X  # input dimension
        self.Y = Y  # output dimension
        self.N = N  # number of memory slot
        self.W = W  # dimension of one memory slot
        self.R = R  # number of read heads
        # DeepLSTM(W * R + X, Y + W * R + 3 * W + 5 * R + 3)
        self.controller = controller

        super(DNC, self).__init__(
            embed_id=L.EmbedID(self.X, self.X),
            l_dl=self.controller,
            l_Wr=L.Linear(self.R * self.W, self.Y, nobias=True)
        )  # <question : should all learnable weights be here??>
        self.reset_state()

    def __call__(self, x):
        xp = self.xp
        batch_size = x.shape[0]
        self.initialize_state(batch_size)
        x = self.embed_id(x)
        self.chi = F.concat((x, self.r))
        (self.nu, self.xi) = \
            F.split_axis(self.l_dl(self.chi), [self.Y], 1)
        (self.kr, self.betar, self.kw, self.betaw,
         self.e, self.v, self.f, self.ga, self.gw, self.pi
         ) = F.split_axis(self.xi, np.cumsum(
             [self.W * self.R, self.R, self.W, 1, self.W, self.W, self.R, 1, 1]), 1)

        self.kr = F.reshape(self.kr, (-1, self.R, self.W))  # _ * R * W
        self.betar = 1 + F.softplus(self.betar)  # _ * R
        # self.kw : _ * W
        self.betaw = 1 + F.softplus(self.betaw)  # _ * 1
        self.e = F.sigmoid(self.e)  # _ * W
        # self.v : _ * W
        self.f = F.sigmoid(self.f)  # _ * R
        self.ga = F.sigmoid(self.ga)  # _ * 1
        self.gw = F.sigmoid(self.gw)  # _ * 1

        # self.p : _ * R * 3
        self.pi = F.reshape(self.pi, (batch_size, self.R, 3))
        self.pi = F.exp(self.pi)
        self.pi = F.scale(self.pi, 1 / F.sum(self.pi, 2), 0)

        # Need https://github.com/pfnet/chainer/issues/1812
        # self.psi : _ * N
        psi_mat = 1 - F.scale(self.wr, self.f, 0)
        self.psi = Variable(xp.ones((batch_size, self.N),
                                    dtype=np.float32), volatile='auto')
        for i in range(self.R):
            self.psi = self.psi * psi_mat[:, i, :]

        self.u = (self.u + self.ww - (self.u * self.ww)) * self.psi  # _ * N

        self.a = u2a(self.u)  # _ * N
        self.cw = C(self.M, self.kw, self.betaw)  # _ * N
        self.ww = F.scale((F.scale(self.a, self.ga, 0) +
                           F.scale(self.cw, (1 - self.ga), 0)), self.gw, 0)  # _ * N
        self.M = self.M * (1.0 - F.batch_matmul(self.ww, self.e, transb=True)
                           ) + F.batch_matmul(self.ww, self.v, transb=True)  # _ * N * W

        # self.L : _ * N * N
        self.L = self.L * (1 - F.broadcast_to(F.reshape(self.ww, (-1, self.N, 1)), (batch_size, self.N, self.N)) - F.broadcast_to(F.reshape(self.ww, (-1, 1,  self.N)), (batch_size, self.N, self.N))) \
            + F.batch_matmul(self.ww, self.p, transb=True)
        self.L = self.L * (xp.ones((self.N, self.N)) -
                           xp.eye(self.N))  # force L[i,i] == 0

        self.p = F.scale(self.p, 1 - F.sum(self.ww, 1), 0) + self.ww  # _ * N

        self.fo = F.transpose(F.batch_matmul(
            self.L, self.wr, transb=True), [0, 2, 1])  # _ * R * N
        self.ba = F.transpose(F.batch_matmul(
            self.L, self.wr, transa=True, transb=True), [0, 2, 1])  # _ * R * N
        self.cr = Cr(self.M, self.kr, self.betar)  # _ * R * N

        self.wr = F.scale(self.ba, self.pi[:, :, 0], 0) + F.scale(
            self.cr, self.pi[:, :, 1], 0) + F.scale(self.fo, self.pi[:, :, 2], 0)  # _ * R * N
        self.r = F.reshape(F.batch_matmul(self.wr, self.M),
                           (batch_size, self.R * self.W))  # _ * RW

        self.y = self.l_Wr(self.r) + self.nu  # _ * Y

        return self.y

    def reset_state(self):
        self.u = self.p = self.L = self.M = self.r = self.wr = self.ww = None
        if hasattr(self.controller, 'reset_state'):
            self.controller.reset_state()
        # any variable else ?

    def initialize_state(self, batch_size):
        xp = self.xp
        state_list = {
            'u': (batch_size, self.N),
            'p': (batch_size, self.N),
            'L': (batch_size, self.N, self.N),
            'M': (batch_size, self.N, self.W),
            'r': (batch_size, self.R * self.W),
            'wr': (batch_size, self.R, self.N),
            'ww': (batch_size, self.N),
        }
        for name, size in state_list.items():
            if getattr(self, name) is None:
                setattr(self, name, Variable(
                    xp.zeros(size, dtype=np.float32), volatile='auto'))


# Simple test code
if __name__ == '__main__':
    X = 5
    Y = 5
    N = 24
    W = 5
    R = 4
    mdl = DNC(X, Y, N, W, R, L.Linear(
        DNC_input_len(X, W, R), DNC_output_len(Y, W, R)))
    xp = np
    opt = optimizers.RMSprop()  # SGD(lr=1.0)
    opt.setup(mdl)
    opt.add_hook(chainer.optimizer.GradientClipping(5))
    datanum = 2000
    loss = 0.0
    acc = 0.0
    for datacnt in range(datanum):
        contentlen = np.random.randint(3, 6)
        content = np.random.randint(1, X, contentlen)
        seqlen = contentlen * 2

        mdl.reset_state()

        loss = 0
        # Remember sequence ...
        for i in range(contentlen):
            x = Variable(xp.array([content[i]], dtype=np.int32))
            mdl(x)
        # Recall !!
        for i in range(contentlen):
            x = Variable(xp.array([0], dtype=np.int32))
            t = Variable(xp.array([content[i]], dtype=np.int32))
            y = mdl(x)
            loss += F.softmax_cross_entropy(y, t)
            print(y.data, t.data, np.argmax(y.data) == content[i])

        loss /= contentlen
        opt.zero_grads()
        loss.backward()
        opt.update()
        loss.unchain_backward()
        print('(', datacnt, ')', loss.data.sum() /
              loss.data.size / contentlen, acc / contentlen)
