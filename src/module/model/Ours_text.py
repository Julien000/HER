import torch
from .Ours import AdaIn,  Our_Model, FusionModel


class FusionModel_Text(FusionModel):
    def __init__(self, emb_size=128, layers=6, feature_num=2, num_heads=4, dropout=0.3):
        super().__init__(emb_size, layers, feature_num, num_heads, dropout)

    def forward(self, text_emb, event_t_emb, flag):
        # 通过 MLP 转换
        text_feature = self.mlp_text(text_emb)
        event_t_feature = self.mlp_event_t(event_t_emb)

        fusion_feature = torch.cat([text_feature.unsqueeze(1), \
                                    event_t_feature.unsqueeze(1)], dim=1)
        for index, transformer in enumerate(self.transformers):
            if index == 3 and flag==1:
                fusion_feature = transformer(fusion_feature, flag=1).squeeze(1) + fusion_feature
            else:
                fusion_feature = transformer(fusion_feature, flag=0).squeeze(1) + fusion_feature
        fusion_mean = fusion_feature.mean(dim=1)  # 均值池化
        fusion_max = fusion_feature.max(dim=1)[0]  # 最大池化
        final_fusion = torch.cat([fusion_mean, fusion_max], dim=-1)  # 拼接两个特征
        # 输出分类结果
        output = self.mlp_out(final_fusion)

        return output


class Our_Model_Text(Our_Model):
    def __init__(self, config, classes=2, p=40):
        super().__init__(config)
        # str_path = './bert-base-english' 
        self.fusion = FusionModel_Text()

    def test_struct(self, text_clip=None, image_clip=None):
        with torch.no_grad():
            text_clip = text_clip.to(torch.float32).squeeze()
            text_embedding = self.txt_encoder(text_clip)

            content_t = self.t_content(text_embedding)
            t_pos, t_neg = self.resolve(text_embedding, content_t)
            domain_t = self.t_domain(t_neg)

            content = self.fusion( content_t,  domain_t, flag=1)

            content_pred = self.CE(content)

            content_i=""
            domain_i=""
            return content_pred, content_t, content, domain_t, content_i, domain_i


    def forward(self,text_clip=None, image_clip=None):
        # text
        text_clip = text_clip.to(torch.float32).squeeze()
        text_embedding = self.txt_encoder(text_clip)

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
        domainOut_i, contentOut_i, content_i, domain_i= "", "", "", ""
        return num, content_pred, content, domainOut_t, contentOut_t, domainOut_i, contentOut_i, content_t, domain_t, content_i, domain_i




