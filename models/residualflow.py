import numpy as np 
import torch
import torch.nn as nn

from iresblock import iResBlock
from container import SequentialFlow
from normalization import MovingBatchNorm1d # 또는 MovingBatchNorm2d
from elemwise import ZeroMeanTransform

class SimpleResidualFlow(nn.Module):
    def __init__(self, input_dim, num_blocks=5, hidden_dim=64, lipschitz_const=0.9):
        super().__init__()
        
        # iResBlock 내부에서 사용할 신경망 (nnet) 정의
        # nnet은 Lipschitz 제약을 위해 get_linear/get_conv2d를 사용할 수도 있습니다.
        # 여기서는 간단히 nn.Sequential로 구성합니다.
        def create_nnet(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(), # 비선형성
                nn.Linear(hidden_dim, out_dim),
            )

        layers = []
        for _ in range(num_blocks):
            # 각 iResBlock에 독립적인 nnet 인스턴스 제공
            nnet_for_block = create_nnet(input_dim, input_dim) 
            layers.append(
                iResBlock(
                    nnet=nnet_for_block,
                    n_power_series=5, # 파워 시리즈 반복 횟수
                    lipschitz_const=lipschitz_const # 립시츠 상수 제약
                )
            )
            # 배치 정규화 추가 (선택 사항이지만 학습 안정성에 도움)
            layers.append(MovingBatchNorm1d(input_dim) if input_dim > 1 else nn.Identity())
            # ZeroMeanTransform 추가 (선택 사항)
            layers.append(ZeroMeanTransform())

        self.flow = SequentialFlow(layers)

    def forward(self, x, logpx=None):
        return self.flow.forward(x, logpx)

    def inverse(self, y, logpy=None):
        return self.flow.inverse(y, logpy)


class ResidualFlow(nn.module):

    def __init__(
        self,
        input_size,
        n_blocks=[16, 16],
        intermediate_dim=64,
        factor_out=True,
        quadratic=False,
        init_layer=None,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
        classification=False,
        classification_hdim=64,
        n_classes=10,
        block_type='resblock',
    ):
        super(ResidualFlow, self).__init__()
        self.n_scale = min(len(n_blocks), self._calc_n_scale(input_size))
        self.n_blocks = n_blocks
        self.intermediate_dim = intermediate_dim
        self.factor_out = factor_out
        self.quadratic = quadratic
        self.init_layer = init_layer
        self.actnorm = actnorm
        self.fc_actnorm = fc_actnorm
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.fc = fc
        self.coeff = coeff
        self.vnorms = vnorms
        self.n_lipschitz_iters = n_lipschitz_iters
        self.sn_atol = sn_atol
        self.sn_rtol = sn_rtol
        self.n_power_series = n_power_series
        self.n_dist = n_dist
        self.n_samples = n_samples
        self.kernels = kernels
        self.activation_fn = activation_fn
        self.fc_end = fc_end
        self.fc_idim = fc_idim
        self.n_exact_terms = n_exact_terms
        self.preact = preact
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.first_resblock = first_resblock
        self.learn_p = learn_p
        self.classification = classification
        self.classification_hdim = classification_hdim
        self.n_classes = n_classes
        self.block_type = block_type


        self.transforms = self._build_net(input_size)
        self.dim = [o[1:] for o in self.calc_output_size(input_size)]

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        if not self.factor_out:
            k = self.n_scale - 1
            return [[n, c * 4**k, h // 2**k, w // 2**k]]
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def _build_net(self,input_size):
        _, c, h, w = input_size
        transforms = []


    def forward(self, x, logpx=None, inverse=False):
        if inverse:
            return self.inverse(x,logx)
        out=[]
        for i in range(len(self.transforms)):
            x,logpx = self.transforms[i].forward(x,logpx)

            #아마도 factor 추출과 관련된 hyperparameter?
            if self.factor_out and (idx < len(self.transforms) - 1):
                d = x.size(1) // 2
                x, f = x[:, :d], x[:, d:]
                out.append(f)
        
        out.append(x)
        out = torch.cat([o.view(o.size()[0], -1) for o in out], 1)
        output = out if logpx is None else (out, logpx)
        return output
    
    
    def inverse(self, z, logpz=None):
        if self.factor_out:
            z = z.view(z.shape[0], -1)
            zs = []
            i = 0
            for dims in self.dims:
                s = np.prod(dims)
                zs.append(z[:, i:i + s])
                i += s
            zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]

            if logpz is None:
                z_prev = self.transforms[-1].inverse(zs[-1])
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev = self.transforms[idx].inverse(z_prev)
                return z_prev
            else:
                z_prev, logpz = self.transforms[-1].inverse(zs[-1], logpz)
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev, logpz = self.transforms[idx].inverse(z_prev, logpz)
                return z_prev, logpz
        else:
            z = z.view(z.shape[0], *self.dims[-1])
            for idx in range(len(self.transforms) - 1, -1, -1):
                if logpz is None:
                    z = self.transforms[idx].inverse(z)
                else:
                    z, logpz = self.transforms[idx].inverse(z, logpz)
            return z if logpz is None else (z, logpz)
        

class FCWrapper(nn.Module):

    def __init__(self, fc_module):
        super(FCWrapper, self).__init__()
        self.fc_module = fc_module

    def forward(self, x, logpx=None):
        shape = x.shape
        x = x.view(x.shape[0], -1)
        if logpx is None:
            y = self.fc_module(x)
            return y.view(*shape)
        else:
            y, logpy = self.fc_module(x, logpx)
            return y.view(*shape), logpy

    def inverse(self, y, logpy=None):
        shape = y.shape
        y = y.view(y.shape[0], -1)
        if logpy is None:
            x = self.fc_module.inverse(y)
            return x.view(*shape)
        else:
            x, logpx = self.fc_module.inverse(y, logpy)
            return x.view(*shape), logpx



class StackediResBlocks(layers.SequentialFlow):

    def __init__(
        self,
        initial_size,
        idim,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
    ):

        chain = []

        # Parse vnorms
        ps = []
        for p in vnorms:
            if p == 'f':
                ps.append(float('inf'))
            else:
                ps.append(float(p))
        domains, codomains = ps[:-1], ps[1:]
        assert len(domains) == len(kernels.split('-'))

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return layers.ActNorm2d(size[0])

        def _quadratic_layer(initial_size, fc):
            if fc:
                c, h, w = initial_size
                dim = c * h * w
                return FCWrapper(layers.InvertibleLinear(dim))
            else:
                return layers.InvertibleConv2d(initial_size[0])

        def _lipschitz_layer(fc):
            return base_layers.get_linear if fc else base_layers.get_conv2d

        def _resblock(initial_size, fc, idim=idim, first_resblock=False):
            if fc:
                return layers.iResBlock(
                    FCNet(
                        input_shape=initial_size,
                        idim=idim,
                        lipschitz_layer=_lipschitz_layer(True),
                        nhidden=len(kernels.split('-')) - 1,
                        coeff=coeff,
                        domains=domains,
                        codomains=codomains,
                        n_iterations=n_lipschitz_iters,
                        activation_fn=activation_fn,
                        preact=preact,
                        dropout=dropout,
                        sn_atol=sn_atol,
                        sn_rtol=sn_rtol,
                        learn_p=learn_p,
                    ),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )
            else:
                ks = list(map(int, kernels.split('-')))
                if learn_p:
                    _domains = [nn.Parameter(torch.tensor(0.)) for _ in range(len(ks))]
                    _codomains = _domains[1:] + [_domains[0]]
                else:
                    _domains = domains
                    _codomains = codomains
                nnet = []
                if not first_resblock and preact:
                    if batchnorm: nnet.append(layers.MovingBatchNorm2d(initial_size[0]))
                    nnet.append(ACT_FNS[activation_fn](False))
                nnet.append(
                    _lipschitz_layer(fc)(
                        initial_size[0], idim, ks[0], 1, ks[0] // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                        domain=_domains[0], codomain=_codomains[0], atol=sn_atol, rtol=sn_rtol
                    )
                )
                if batchnorm: nnet.append(layers.MovingBatchNorm2d(idim))
                nnet.append(ACT_FNS[activation_fn](True))
                for i, k in enumerate(ks[1:-1]):
                    nnet.append(
                        _lipschitz_layer(fc)(
                            idim, idim, k, 1, k // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                            domain=_domains[i + 1], codomain=_codomains[i + 1], atol=sn_atol, rtol=sn_rtol
                        )
                    )
                    if batchnorm: nnet.append(layers.MovingBatchNorm2d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))
                if dropout: nnet.append(nn.Dropout2d(dropout, inplace=True))
                nnet.append(
                    _lipschitz_layer(fc)(
                        idim, initial_size[0], ks[-1], 1, ks[-1] // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                        domain=_domains[-1], codomain=_codomains[-1], atol=sn_atol, rtol=sn_rtol
                    )
                )
                if batchnorm: nnet.append(layers.MovingBatchNorm2d(initial_size[0]))
                return layers.iResBlock(
                    nn.Sequential(*nnet),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )

        if init_layer is not None: chain.append(init_layer)
        if first_resblock and actnorm: chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm: chain.append(_actnorm(initial_size, True))

        if squeeze:
            c, h, w = initial_size
            for i in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, first_resblock=first_resblock and (i == 0)))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            chain.append(layers.SqueezeLayer(2))
        else:
            for _ in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True, fc_idim))
                    if actnorm or fc_actnorm: chain.append(_actnorm(initial_size, True))

        super(StackediResBlocks, self).__init__(chain)


class FCNet(nn.Module):

    def __init__(
        self, input_shape, idim, lipschitz_layer, nhidden, coeff, domains, codomains, n_iterations, activation_fn,
        preact, dropout, sn_atol, sn_rtol, learn_p, div_in=1
    ):
        super(FCNet, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape
        dim = c * h * w
        nnet = []
        last_dim = dim // div_in
        if preact: nnet.append(ACT_FNS[activation_fn](False))
        if learn_p:
            domains = [nn.Parameter(torch.tensor(0.)) for _ in range(len(domains))]
            codomains = domains[1:] + [domains[0]]
        for i in range(nhidden):
            nnet.append(
                lipschitz_layer(last_dim, idim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                    last_dim, idim, coeff=coeff, n_iterations=n_iterations, domain=domains[i], codomain=codomains[i],
                    atol=sn_atol, rtol=sn_rtol
                )
            )
            nnet.append(ACT_FNS[activation_fn](True))
            last_dim = idim
        if dropout: nnet.append(nn.Dropout(dropout, inplace=True))
        nnet.append(
            lipschitz_layer(last_dim, dim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                last_dim, dim, coeff=coeff, n_iterations=n_iterations, domain=domains[-1], codomain=codomains[-1],
                atol=sn_atol, rtol=sn_rtol
            )
        )
        self.nnet = nn.Sequential(*nnet)