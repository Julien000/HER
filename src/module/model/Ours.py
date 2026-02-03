import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from datetime import datetime

from .orthogonality import selfAttention

data_idx=[]

class CrossModule4Batch(nn.Module):
    def __init__(self, text_in_dim=128, image_in_dim=128, corre_out_dim=128):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 128
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_in = text.unsqueeze(2)
        image_in = image.unsqueeze(1)
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        correlation_out = self.c_specific_2(correlation_p)
        return correlation_out

def AdaIn(content, style, style_strength=0.1, eps=1e-5):

    content_std, content_mean = torch.std_mean(content, dim=-1, unbiased=False, keepdim=True)
    style_std, style_mean = torch.std_mean(style, dim=-1, unbiased=False, keepdim=True)
    normalized_content = (content - content_mean) / (content_std + eps)
    stylized_content = (normalized_content * style_std) + style_mean
    output = (1 - style_strength) * content + style_strength * stylized_content
    return output

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=2, dropout=0, initializer_range=0.02):
        super(MultiHeadSelfAttention, self).__init__()
        self.config = list()

        self.hidden_size = hidden_size

        self.head_num = head_num
        if (self.hidden_size) % head_num != 0:
            raise ValueError(self.head_num, "error")
        self.head_dim = self.hidden_size // self.head_num

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.concat_weight = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        torch.nn.init.normal_(self.query.weight, 0, initializer_range)
        torch.nn.init.normal_(self.key.weight, 0, initializer_range)
        torch.nn.init.normal_(self.value.weight, 0, initializer_range)
        torch.nn.init.normal_(self.concat_weight.weight, 0, initializer_range)
        self.dropout = torch.nn.Dropout(dropout)
        self.idx_counts=0

    def dot_score(self, encoder_output):
        query = self.dropout(self.query(encoder_output))
        key = self.dropout(self.key(encoder_output))
        # head_num * batch_size * session_length * head_dim
        querys = torch.stack(query.chunk(self.head_num, -1), 0)
        keys = torch.stack(key.chunk(self.head_num, -1), 0)
        # head_num * batch_size * session_length * session_length
        dots = querys.matmul(keys.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        # print(len(dots),dots[0].shape)
        return dots

    def forward(self, encoder_outputs, mask=None, flag=0): 
        attention_energies = self.dot_score(encoder_outputs)
        value = self.dropout(self.value(encoder_outputs))

        values = torch.stack(value.chunk(self.head_num, -1))

        if mask is not None:
            eye = torch.eye(mask.shape[-1]).to('cuda')
            new_mask = torch.clamp_max((1 - (1 - mask.float()).unsqueeze(1).permute(0, 2, 1).bmm(
                (1 - mask.float()).unsqueeze(1))) + eye, 1)
            attention_energies = attention_energies - new_mask * 1e12
            weights = F.softmax(attention_energies, dim=-1)
            weights = weights * (1 - new_mask)
        else:
            weights = F.softmax(attention_energies, dim=2)
        x = weights[0]

        batch_size = x.shape[0]
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if flag == 1:# pass

            summed = torch.sum(x, dim=2)  # 在列的维度上求和，得到batch*4的张量
            # 对4*1的张量进行softmax操作 
            softmaxed = F.softmax(summed, dim=1)  # 在行的维度上应用softmax 
            softmaxed_np = softmaxed.detach().cpu().numpy() 
            #TODO 输出 # print((softmaxed_np))
            # plot_attention_heatmap( softmaxed_np, output_dir=f"workdir/hm/{current_time}_HM.png")
            # Step 1: 平均所有 head 和 batch
            self.idx_counts = self.idx_counts+batch_size

        outputs = weights.matmul(values)
        outputs = torch.cat([outputs[i] for i in range(outputs.shape[0])], dim=-1)
        outputs = self.dropout(self.concat_weight(outputs))

        return outputs

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, initializer_range=0.02):
        super(PositionWiseFeedForward, self).__init__()
        self.final1 = torch.nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.final2 = torch.nn.Linear(hidden_size * 4, hidden_size, bias=True)
        torch.nn.init.normal_(self.final1.weight, 0, initializer_range)
        torch.nn.init.normal_(self.final2.weight, 0, initializer_range)

    def forward(self, x):
        x = F.relu(self.final1(x))
        x = self.final2(x)
        return x

class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=4, dropout=0, attention_dropout=0,
                 initializer_range=0.02):
        super(TransformerLayer, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.mh = MultiHeadSelfAttention(hidden_size=hidden_size, activate=activate, head_num=head_num,
                                         dropout=attention_dropout, initializer_range=initializer_range)
        self.pffn = PositionWiseFeedForward(hidden_size, initializer_range=initializer_range)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, encoder_outputs, mask=None, flag=0):
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.mh(encoder_outputs, mask, flag)))
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.pffn(encoder_outputs)))
        return encoder_outputs

class MLP_trans(torch.nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLP_trans, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_1 = nn.Linear(input_size, out_size)
        # self.mlp_2 = nn.Linear(out_size, out_size)


    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        # emb_trans = self.dropout(self.activate(self.mlp_2(emb_trans)))

        return emb_trans

class MLP_merge_star(torch.nn.Module):
    def __init__(self, imput_size, out_size, dropout=0.2):
        super(MLP_merge_star, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_s = nn.Linear(imput_size, int(imput_size / 2))

    def forward(self, emb_trans):
        results = self.dropout(self.activate(self.mlp_s(emb_trans)))
        return results

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FusionModel(nn.Module):
    #原始：layers=4, num_heads=4, dropout=0.3 0.73
    # layers = 5, num_heads = 4, dropout = 0.3 0.7375
    # layers = 3, num_heads = 4, dropout = 0.3 0.7366

    # layers = 6, num_heads = 4, dropout = 0.3 0.7400, 用平均的方法 sentence=8
    def __init__(self, emb_size=128, layers=6, feature_num=2, num_heads=4, dropout=0.3):
        super(FusionModel, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.mlp_text = MLP_trans(emb_size, emb_size, dropout=dropout)
        self.mlp_img = MLP_trans(emb_size, emb_size, dropout=dropout)
        self.mlp_event_t = MLP_trans(emb_size, emb_size, dropout=dropout)
        self.mlp_event_i = MLP_trans(emb_size, emb_size, dropout=dropout)

        self.transformers = nn.ModuleList([TransformerLayer(emb_size, head_num=num_heads, dropout=0.3, attention_dropout=0, initializer_range=0.02) for _ in range(layers)])

        self.layer_norm = nn.LayerNorm(emb_size)


        self.mlp_out = nn.Sequential(
            nn.Linear(emb_size * 2, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, image_emb, text_emb, event_i_emb, event_t_emb, flag):

        # 通过 MLP 转换
        img_feature = self.mlp_img(image_emb)
        text_feature = self.mlp_text(text_emb)
        event_i_feature = self.mlp_event_i(event_i_emb)
        event_t_feature = self.mlp_event_t(event_t_emb)

        fusion_feature = torch.cat([img_feature.unsqueeze(1), text_feature.unsqueeze(1), \
                                    event_i_feature.unsqueeze(1), event_t_feature.unsqueeze(1)], dim=1)
        for index, transformer in enumerate(self.transformers):
            # if index == 3 and flag==1:
            #     fusion_feature = transformer(fusion_feature, flag=1).squeeze(1) + fusion_feature
            # 只有在最后一层的时候需要去做可视化
            if index == 3:
                fusion_feature = transformer(fusion_feature, flag=1).squeeze(1) + fusion_feature
            else:
                fusion_feature = transformer(fusion_feature, flag=0).squeeze(1) + fusion_feature


        fusion_mean = fusion_feature.mean(dim=1)  # 均值池化
        fusion_max = fusion_feature.max(dim=1)[0]  # 最大池化
        final_fusion = torch.cat([fusion_mean, fusion_max], dim=-1)  # 拼接两个特征

        # 输出分类结果
        output = self.mlp_out(final_fusion)

        return output
    
class FusionModel_text(nn.Module):
    #原始：layers=4, num_heads=4, dropout=0.3 0.73
    # layers = 5, num_heads = 4, dropout = 0.3 0.7375
    # layers = 3, num_heads = 4, dropout = 0.3 0.7366

    # layers = 6, num_heads = 4, dropout = 0.3 0.7400, 用平均的方法 sentence=8
    def __init__(self, emb_size=128, layers=6, feature_num=2, num_heads=4, dropout=0.3):
        super(FusionModel_text, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads


        self.mlp_text = MLP_trans(emb_size, emb_size, dropout=dropout)
        self.mlp_img = MLP_trans(emb_size, emb_size, dropout=dropout)
        self.mlp_event_t = MLP_trans(emb_size, emb_size, dropout=dropout)
        self.mlp_event_i = MLP_trans(emb_size, emb_size, dropout=dropout)




        self.transformers = nn.ModuleList([TransformerLayer(emb_size, head_num=num_heads, dropout=0.3,
                                                            attention_dropout=0, initializer_range=0.02) for _ in
                                           range(layers)])



        self.layer_norm = nn.LayerNorm(emb_size)


        self.mlp_out = nn.Sequential(
            nn.Linear(emb_size * 2, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self,  text_emb,  event_t_emb, flag):

        # 通过 MLP 转换 
        text_feature = self.mlp_text(text_emb) 
        event_t_feature = self.mlp_event_t(event_t_emb)

        fusion_feature = torch.cat([ text_feature.unsqueeze(1), event_t_feature.unsqueeze(1)], dim=1)
        for index, transformer in enumerate(self.transformers):
            if index == 3 and flag==1:
                fusion_feature = transformer(fusion_feature, flag=1).squeeze(1) + fusion_feature
            else:
                fusion_feature = transformer(fusion_feature, flag=0).squeeze(1) + fusion_feature
        # for transformer in self.transformers:
        #     if transformer != 0:
        #         flag = 0
        #     fusion_feature = transformer(fusion_feature, flag=flag).squeeze(1) + fusion_feature

        fusion_mean = fusion_feature.mean(dim=1)  # 均值池化
        fusion_max = fusion_feature.max(dim=1)[0]  # 最大池化
        final_fusion = torch.cat([fusion_mean, fusion_max], dim=-1)  # 拼接两个特征

        # 输出分类结果
        output = self.mlp_out(final_fusion)

        return output

# class Multi_Model(nn.Module):
class Our_Model(nn.Module):
    def __init__(self, config, classes=2, p=40):
        super(Our_Model, self).__init__()
        # str_path = './bert-base-english'
        self.fusion = FusionModel()
        self.resolve = selfAttention()

        self.config = config
        self.event_num= config.event_num

        self.txt_encoder = nn.Sequential()
        self.txt_encoder.add_module('txt_fc1', nn.Linear(512, 256))
        self.txt_encoder.add_module('txt_dropout', nn.Dropout(p=0.5))
        self.txt_encoder.add_module('txt_fc2', nn.Linear(256, 128))

        self.img_encoder = nn.Sequential()
        self.img_encoder.add_module('img_fc1', nn.Linear(512, 256))
        self.img_encoder.add_module('img_dropout', nn.Dropout(p=0.5))
        self.img_encoder.add_module('img_fc2', nn.Linear(256, 128))

        #class classifier
        self.CE = nn.Sequential()
        self.CE.add_module('ce_fc1', nn.Linear(128, 64))
        self.CE.add_module('ce_dropout', nn.Dropout(p=0.5))
        self.CE.add_module('ce_fc2', nn.Linear(64, 2))

        self.CE_i = nn.Sequential()
        self.CE_i.add_module('cei_fc1', nn.Linear(128, 2))
        # self.CE_i.add_module('cei_dropout', nn.Dropout(p=0.5))
        # self.CE_i.add_module('cei_fc2', nn.Linear(64, 2))

        self.CE_t = nn.Sequential()
        self.CE_t.add_module('cet_fc1', nn.Linear(128, 2))
        # self.CE_t.add_module('cet_dropout', nn.Dropout(p=0.5))
        # self.CE_t.add_module('cet_fc2', nn.Linear(64, 2))

        # domain classifier
        self.EE_t = nn.Sequential()
        self.EE_t.add_module('det_fc1', nn.Linear(128, self.event_num))
        # self.EE_t.add_module('det_dropout', nn.Dropout(p=0.5))
        # self.EE_t.add_module('det_fc2', nn.Linear(64, 12))
        #
        self.EE_i = nn.Sequential()
        self.EE_i.add_module('dei_fc1', nn.Linear(128, self.event_num))
        # self.EE_i.add_module('dei_dropout', nn.Dropout(p=0.3))
        # self.EE_i.add_module('dei_fc2', nn.Linear(64, 12))


        self.i_content = nn.Sequential()
        self.i_content.add_module('i_content_fc1', nn.Linear(128, 128))
        # self.i_content.add_module('i_content_dropout', nn.Dropout(p=0.3))

        self.i_domain = nn.Sequential()
        self.i_domain.add_module('i_domain_fc1', nn.Linear(128, 128))
        # self.i_domain.add_module('i_domain_dropout', nn.Dropout(p=0.3))

        self.t_content = nn.Sequential()
        self.t_content.add_module('t_content_fc1', nn.Linear(128, 128))
        # self.t_content.add_module('t_content_dropout', nn.Dropout(p=0.3))

        self.t_domain = nn.Sequential()
        self.t_domain.add_module('t_domain_fc1', nn.Linear(128, 128))
        # self.t_domain.add_module('t_domain_dropout', nn.Dropout(p=0.3))
        # self.fusion_domain = CrossModule4Batch()

    def adv_params_2(self):
        params = []
        # for m in [self.resolve, self.t_content, self.t_domain, self.txt_encoder, self.img_encoder]:
        for m in [self.txt_encoder, self.img_encoder, \
                  self.resolve, self.t_content, self.t_domain, self.i_content, self.i_domain]:
            params += [p for p in m.parameters()]
        return params
    
    def test_struct(self, text_clip=None, image_clip=None, idx=None):
        if idx!=None:
            data_idx=idx.numpy()

        with torch.no_grad():
            text_clip = text_clip.to(torch.float32).squeeze()
            text_embedding = self.txt_encoder(text_clip)

            content_t = self.t_content(text_embedding)
            t_pos, t_neg = self.resolve(text_embedding, content_t)
            # domain_t = self.t_domain(text_embedding)
            domain_t = self.t_domain(t_neg)

            # image
            image_clip = image_clip.to(torch.float32).squeeze()
            image_embedding = self.img_encoder(image_clip)

            content_i = self.i_content(image_embedding)
            I_pos, I_neg = self.resolve(image_embedding, content_i)
            # domain_i = self.i_domain(image_embedding)
            domain_i = self.i_domain(I_neg)

            content = self.fusion(content_i, content_t, domain_i, domain_t, flag=0) 
            content_pred = self.CE(content)
            return content_pred, content, content_t, domain_t, content_i, domain_i

    def forward(self, text_clip=None, image_clip=None, idx=None):
        if idx!=None:
           data_idx=idx.numpy()
        # text
        text_clip = text_clip.to(torch.float32).squeeze()
        text_embedding = self.txt_encoder(text_clip)

        #self.resolve 用于解耦
        content_t = self.t_content(text_embedding)
        t_pos, t_neg = self.resolve(text_embedding, content_t) 
        domain_t = self.t_domain(t_neg)

        # image
        image_clip = image_clip.to(torch.float32).squeeze()
        image_embedding = self.img_encoder(image_clip)

        content_i = self.i_content(image_embedding)
        I_pos, I_neg = self.resolve(image_embedding, content_i) 
        domain_i = self.i_domain(I_neg)

        #开始挑选扰动sample
        n = content_i.shape[0]
        num = int(n) #随机扰动的个数,原来是1
        indices = torch.stack([torch.randperm(n)[:num] for _ in range(n)])

        domain_t_selected = domain_t[indices]  # (n, 3, 128)
        content_t_expanded = content_t.unsqueeze(1).expand(-1, num, -1)
        content_sum_t = AdaIn(content_t_expanded, domain_t_selected).reshape(n * num, 128)  # (n * 3, 128)

        content_t_selected = content_t[indices]
        domain_t_expanded = domain_t.unsqueeze(1).expand(-1, num, -1)
        domain_sum_t = AdaIn(domain_t_expanded, content_t_selected).reshape(n * num, 128)
        #
        domainOut_t = self.EE_t(domain_sum_t)
        contentOut_t = self.CE_t(content_sum_t)

        domain_i_selected = domain_i[indices]  # (n, 3, 128)
        content_i_expanded = content_i.unsqueeze(1).expand(-1, num, -1)
        content_sum_i = AdaIn(content_i_expanded, domain_i_selected).reshape(n * num, 128)  # (n * 3, 128)
        #
        content_i_selected = content_i[indices]
        domain_i_expanded = domain_i.unsqueeze(1).expand(-1, num, -1)
        domain_sum_i = AdaIn(domain_i_expanded, content_i_selected).reshape(n * num, 128)
        #
        domainOut_i = self.EE_i(domain_sum_i)
        contentOut_i = self.CE_i(content_sum_i)
 
        # content = self.fusion(content_i, content_t, domain_i, domain_t, flag=0)
        content = self.fusion(content_i, content_t, domain_i, domain_t, flag=1)
        content_pred = self.CE(content)

        return num, content_pred, content, domainOut_t, contentOut_t, domainOut_i, contentOut_i, content_t, domain_t, content_i, domain_i
    
class Our_Model_text(nn.Module):
    def __init__(self, config, classes=2, p=40):
        super(Our_Model_text, self).__init__()
        # str_path = './bert-base-english'
        self.fusion = FusionModel_text()
        self.resolve = selfAttention()

        self.config = config
        self.event_num= config.event_num

        self.txt_encoder = nn.Sequential()
        self.txt_encoder.add_module('txt_fc1', nn.Linear(512, 256))
        self.txt_encoder.add_module('txt_dropout', nn.Dropout(p=0.5))
        self.txt_encoder.add_module('txt_fc2', nn.Linear(256, 128))

 

        #class classifier
        self.CE = nn.Sequential()
        self.CE.add_module('ce_fc1', nn.Linear(128, 64))
        self.CE.add_module('ce_dropout', nn.Dropout(p=0.5))
        self.CE.add_module('ce_fc2', nn.Linear(64, 2))


        self.CE_t = nn.Sequential()
        self.CE_t.add_module('cet_fc1', nn.Linear(128, 2))
        # self.CE_t.add_module('cet_dropout', nn.Dropout(p=0.5))
        # self.CE_t.add_module('cet_fc2', nn.Linear(64, 2))

        # domain classifier
        self.EE_t = nn.Sequential()
        self.EE_t.add_module('det_fc1', nn.Linear(128, self.event_num))
        # self.EE_t.add_module('det_dropout', nn.Dropout(p=0.5))
        # self.EE_t.add_module('det_fc2', nn.Linear(64, 12))
        #

        self.t_content = nn.Sequential()
        self.t_content.add_module('t_content_fc1', nn.Linear(128, 128))
        # self.t_content.add_module('t_content_dropout', nn.Dropout(p=0.3))

        self.t_domain = nn.Sequential()
        self.t_domain.add_module('t_domain_fc1', nn.Linear(128, 128))
        # self.t_domain.add_module('t_domain_dropout', nn.Dropout(p=0.3))
        # self.fusion_domain = CrossModule4Batch()

    def adv_params_2(self):
        params = []
        # for m in [self.resolve, self.t_content, self.t_domain, self.txt_encoder, self.img_encoder]:
        for m in [self.txt_encoder, self.resolve, self.t_content, self.t_domain]:
            params += [p for p in m.parameters()]
        return params
    
    def test_struct(self, text_clip=None, image_clip=None):
        with torch.no_grad():
            text_clip = text_clip.to(torch.float32).squeeze()
            text_embedding = self.txt_encoder(text_clip)

            content_t = self.t_content(text_embedding)
            t_pos, t_neg = self.resolve(text_embedding, content_t)
            # domain_t = self.t_domain(text_embedding)
            domain_t = self.t_domain(t_neg)

            
            content = self.fusion( content_t, domain_t, flag=1) 
            content_pred = self.CE(content)
            return content_pred, content, content_t, domain_t, '', ''

    def forward(self, text_clip=None, image_clip=None):
        # text
        text_clip = text_clip.to(torch.float32).squeeze()
        text_embedding = self.txt_encoder(text_clip)

        #self.resolve 用于解耦
        content_t = self.t_content(text_embedding)
        t_pos, t_neg = self.resolve(text_embedding, content_t) 
        domain_t = self.t_domain(t_neg)


        #开始挑选扰动sample
        n = content_t.shape[0]
        num = int(n) #随机扰动的个数,原来是1
        indices = torch.stack([torch.randperm(n)[:num] for _ in range(n)])


        domain_t_selected = domain_t[indices]  # (n, 3, 128)
        content_t_expanded = content_t.unsqueeze(1).expand(-1, num, -1)
        content_sum_t = AdaIn(content_t_expanded, domain_t_selected).reshape(n * num, 128)  # (n * 3, 128)

        content_t_selected = content_t[indices]
        domain_t_expanded = domain_t.unsqueeze(1).expand(-1, num, -1)
        domain_sum_t = AdaIn(domain_t_expanded, content_t_selected).reshape(n * num, 128)
        #
        domainOut_t = self.EE_t(domain_sum_t)
        contentOut_t = self.CE_t(content_sum_t)


 
        content = self.fusion(content_t, domain_t, flag=0)
        content_pred = self.CE(content)

        return num, content_pred, content, domainOut_t, contentOut_t, '', ' ', content_t, domain_t, '', ''




