import torch.nn as nn
import torch

class Encoder(nn.Module) :
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_input, n_z, act_f='relu') :
        super(Encoder, self).__init__()
        self.Act_func = {
                'relu' : nn.ReLU,
                'prelu' : nn.PReLU,
                'elu' : nn.ELU,
                'leakyrelu' : nn.LeakyReLU,
                'selu' : nn.SELU,
                }
        self.Encder1 = nn.Sequential \
                    (
                    nn.Linear(n_input, n_enc_1),
                    nn.BatchNorm1d(n_enc_1),
                    self.Act_func[act_f](),
                    nn.Dropout(),
                    nn.Linear(n_enc_1, n_enc_2),
                    nn.BatchNorm1d(n_enc_2),
                    self.Act_func[act_f](),
                    nn.Dropout(),
                    nn.Linear(n_enc_2, n_enc_3),
                    nn.BatchNorm1d(n_enc_3),
                    self.Act_func[act_f](),
                    nn.Dropout(),
                    nn.Linear(n_enc_3, n_z),
                    )

    def forward(self, x) :
        z1 = self.Encder1(x.float())

        return z1

class Decoder(nn.Module) :
    def __init__(self, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, act_f='relu') :
        super(Decoder, self).__init__()
        self.Act_func = {
                'relu' : nn.ReLU,
                'prelu' : nn.PReLU,
                'elu' : nn.ELU,
                'leakyrelu' : nn.LeakyReLU,
                'selu' : nn.SELU,
                }

        self.Decoder1 = nn.Sequential \
                    (
                    nn.Linear(n_z, n_dec_1),
                    nn.BatchNorm1d(n_dec_1),
                    self.Act_func[act_f](),
                    nn.Dropout(),

                    nn.Linear(n_dec_1, n_dec_2),
                    nn.BatchNorm1d(n_dec_2),
                    self.Act_func[act_f](),
                    nn.Dropout(),
                    nn.Linear(n_dec_2, n_dec_3),
                    nn.BatchNorm1d(n_dec_3),
                    self.Act_func[act_f](),
                    nn.Dropout(),
                    nn.Linear(n_dec_3, n_input),
                    )

    def forward(self, z) :
        x_bar = self.Decoder1(z)

        return x_bar

class HGLGNN(nn.Module) :
    def __init__(
            self, dims_X, dims_A, n_enc_1=500, n_enc_2=1000, n_enc_3=500, n_dec_1=500, n_dec_2=1000,
            n_dec_3=500, n_z=64, act_f='relu'
            ) -> None :
        super().__init__()

        self.encoder1 = Encoder(n_enc_1, n_enc_2, n_enc_3, dims_X, n_z, act_f)
        self.encoder2 = Encoder(n_enc_1, n_enc_2, n_enc_3, dims_A, n_z, act_f)
        self.decoder1 = Decoder(n_dec_1, n_dec_2, n_dec_3, dims_X, n_z * 2, act_f)


    def reset_parameters(self) :
        self.encoder1.reset_parameters()
        self.encoder2.reset_parameters()
        self.decoder.reset_parameters()

    def dual_encoder(self, A, X) :
        z_x = self.encoder1(X)
        z_A = self.encoder2(A)

        return z_x, z_A


    def forward(self, X, A) :
        HX, HA = self.dual_encoder(A, X)
        H = torch.cat((HX, HA))
        Xbar = self.decoder1(H)

        return H, HX, HA, Xbar

