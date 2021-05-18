# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from models.modeling_bertnewsinglecut import BertModelNew
from apex import amp


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def l1norm(X, dim, eps=1e-5):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-5):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).add(eps).sqrt() + eps
    X = torch.div(X, norm)
    return X



class EncoderText(nn.Module):

    def __init__(self,opt):
        super(EncoderText, self).__init__()
        self.MARGIN = opt.margin
        self.MODE = opt.MODE

        model_path = 'bert/'
        self.length = 32

        if self.MODE == 'inflate' or self.MODE == 'shrinktobase':
            self.poly_length = 32
            self.poly_code_embeddings_v = nn.Embedding(self.poly_length, 768)
            self.poly_code_embeddings_t = nn.Embedding(self.poly_length, 768)

        self.encoder = BertModelNew.from_pretrained(model_path)
        
        if self.MODE == 'shrinktobase':
            self.encoder2 = BertModelNew.from_pretrained(model_path)
            self.T_length = 32
            self.S_length = 32
            self.T_w = 16
            self.S_w = 16
            self.poly_length = 32
            self.T_length = 32 + self.poly_length

        elif self.MODE == 'shrinktofast':
            self.encoder2 = BertModelNew.from_pretrained(model_path)
            self.encoder3 = BertModelNew.from_pretrained(model_path)
            self.T_length = 32
            self.S_length = 1
            self.T_w = 2
            self.S_w = 16
            self.T_length = 32 
  
        self.scale = 768
        self.fc = nn.Linear(2048,768)
        self.fc3= nn.Linear(768,1601) 
        self.norm = nn.LayerNorm(768,eps=1e-5)
        self.relu = nn.ReLU()
        self.mlm = opt.mlm
        self.cm = opt.cm
        self.mrm = opt.mrm
        self.fc2 = nn.Linear(768, 1)


    def compute_distill(self, scores_gt,scores_pre):
        scale1 = self.T_w
        scale2 = self.S_w
        cost_gt1 = F.softmax(scores_gt.mul(scale1),dim=1).detach()
        cost_pre1 = F.softmax(scores_pre.mul(scale2),dim=1)
        cost_gt2 = F.softmax(scores_gt.mul(scale1),dim=0).detach()
        cost_pre2 = F.softmax(scores_pre.mul(scale2),dim=0)
        return cost_gt1.mul(cost_pre1.log()).sum().mul(-0.5) + cost_gt2.mul(cost_pre2.log()).sum().mul(-0.5)

    def forward(self, input_ids, token_type_ids, non_pad_mask, vision_feat, vision_mask, gt_labels=None, vision_labels=None, MLM=False, istest=False):
        text_output = self.encoder.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids.long().squeeze())

        head_mask = [None]*20
        vision_feat = self.fc(vision_feat)
        vision_feat = self.norm(vision_feat)
        
        bs = text_output.size(0)
        tl = text_output.size(1)
        vl = vision_feat.size(1)

        if self.MODE == 'inflate':
            poly_code_ids = torch.arange(self.poly_length, dtype=torch.long).cuda()
            poly_code_text = torch.arange(self.poly_length, dtype=torch.long).cuda()
            poly_mask = torch.ones(bs,self.poly_length).long().cuda()
            poly_mask_text = torch.ones(bs,self.poly_length).long().cuda()
            poly_code_ids = poly_code_ids.unsqueeze(0).expand(bs, self.poly_length)
            poly_code_text = poly_code_text.unsqueeze(0).expand(bs, self.poly_length)
            poly_codes = self.poly_code_embeddings_v(poly_code_ids)
            poly_codes_vision = self.norm(poly_codes)
            poly_codes_text = self.poly_code_embeddings_t(poly_code_text)
            poly_codes_text = self.norm(poly_codes_text)
            non_pad_mask = torch.cat([non_pad_mask.long(), poly_mask_text], dim=1)
            vision_mask = torch.cat([vision_mask.long(), poly_mask], dim=1)
            vision_feat = torch.cat([vision_feat, poly_codes_vision], dim=1)
            text_output = torch.cat([text_output, poly_codes_text], dim=1)
            tl = text_output.size(1)
            vl = vision_feat.size(1)
        if self.MODE == 'shrinktobase':
            poly_code_ids = torch.arange(self.poly_length, dtype=torch.long).cuda()
            poly_code_text = torch.arange(self.poly_length, dtype=torch.long).cuda()
            poly_mask = torch.ones(bs,self.poly_length).long().cuda()
            poly_mask_text = torch.ones(bs,self.poly_length).long().cuda()
            poly_code_ids = poly_code_ids.unsqueeze(0).expand(bs, self.poly_length)
            poly_code_text = poly_code_text.unsqueeze(0).expand(bs, self.poly_length)
            poly_codes = self.poly_code_embeddings_v(poly_code_ids)
            poly_codes_vision = self.norm(poly_codes)
            poly_codes_text = self.poly_code_embeddings_t(poly_code_text)
            poly_codes_text = self.norm(poly_codes_text)

            non_pad_mask_plus = torch.cat([non_pad_mask.long(), poly_mask_text], dim=1)
            vision_mask_plus = torch.cat([vision_mask.long(), poly_mask], dim=1)
            vision_feat_plus = torch.cat([vision_feat, poly_codes_vision], dim=1)
            text_output_plus = torch.cat([text_output, poly_codes_text], dim=1)
            tl = text_output_plus.size(1)
            vl = vision_feat_plus.size(1)
            extended_attention_mask_text = non_pad_mask.squeeze()[:, None, None, :]
            extended_attention_mask_text = (1.0 - extended_attention_mask_text) * -10000.0
            extended_attention_mask_vision = vision_mask.squeeze()[:, None, None, :]
            extended_attention_mask_vision = (1.0 - extended_attention_mask_vision) * -10000.0


        if self.MODE == 'base' or self.MODE == 'fast' or self.MODE == 'inflate':
            extended_attention_mask_text = non_pad_mask.squeeze()[:, None, None, :]
            extended_attention_mask_text = (1.0 - extended_attention_mask_text) * -10000.0
            extended_attention_mask_vision = vision_mask.squeeze()[:, None, None, :]
            extended_attention_mask_vision = (1.0 - extended_attention_mask_vision) * -10000.0
            textnew1 = self.encoder.encoder(text_output,extended_attention_mask_text,head_mask)
            textnew1 = textnew1[0]

            visionnew1 = self.encoder.encoder(vision_feat,extended_attention_mask_vision,head_mask)
            visionnew1 = visionnew1[0]

            visionnew1 = F.normalize(visionnew1, p=2, dim=2)
            textnew1 = F.normalize(textnew1, p=2, dim=2)
            
            if self.MODE == 'inflate':
                mini_tl = self.length + self.poly_length
            elif self.MODE == 'fast':
                mini_tl =  1
            else:
                mini_tl = self.length

            visionnew1 = visionnew1[:, :mini_tl]
            textnew1 = textnew1[:, :mini_tl]

            if istest == False:
                textnew1 = textnew1.unsqueeze(0).expand(bs,-1,-1,-1).contiguous().view(bs*bs,mini_tl,-1)
                visionnew1 = visionnew1.unsqueeze(1).expand(-1,bs,-1,-1).contiguous().view(bs*bs,mini_tl,-1)
                
        if self.MODE == 'shrinktobase':
            extended_attention_mask_text = non_pad_mask.squeeze()[:, None, None, :]
            extended_attention_mask_text = (1.0 - extended_attention_mask_text) * -10000.0
            extended_attention_mask_vision = vision_mask.squeeze()[:, None, None, :]
            extended_attention_mask_vision = (1.0 - extended_attention_mask_vision) * -10000.0
            extended_attention_mask_text_plus = non_pad_mask_plus.squeeze()[:, None, None, :]
            extended_attention_mask_text_plus = (1.0 - extended_attention_mask_text_plus) * -10000.0
            extended_attention_mask_vision_plus = vision_mask_plus.squeeze()[:, None, None, :]
            extended_attention_mask_vision_plus = (1.0 - extended_attention_mask_vision_plus) * -10000.0
            textnew1 = self.encoder.encoder(text_output_plus,extended_attention_mask_text_plus,head_mask)
            visionnew1 = self.encoder.encoder(vision_feat_plus,extended_attention_mask_vision_plus,head_mask)
            textnew1 = textnew1[0]
            visionnew1 = visionnew1[0]
            visionnew1 = F.normalize(visionnew1, p=2, dim=2)
            textnew1 = F.normalize(textnew1, p=2, dim=2)
            mini_tl = self.T_length
            visionnew1 = visionnew1[:, :mini_tl]
            textnew1 = textnew1[:, :mini_tl]

            textnew2 = self.encoder2.encoder(text_output,extended_attention_mask_text,head_mask)
            visionnew2 = self.encoder2.encoder(vision_feat,extended_attention_mask_vision,head_mask)
            textnew2 = textnew2[0]
            visionnew2 = visionnew2[0]
            visionnew2 = F.normalize(visionnew2, p=2, dim=2)
            textnew2 = F.normalize(textnew2, p=2, dim=2)
            mini_tl2 = self.S_length
            visionnew2 = visionnew2[:, :mini_tl2]
            textnew2 = textnew2[:, :mini_tl2]
            if istest == False:
                textnew1 = textnew1.unsqueeze(0).expand(bs,-1,-1,-1).contiguous().view(bs*bs,mini_tl,-1)
                visionnew1 = visionnew1.unsqueeze(1).expand(-1,bs,-1,-1).contiguous().view(bs*bs,mini_tl,-1)

                textnew2 = textnew2.unsqueeze(0).expand(bs,-1,-1,-1).contiguous().view(bs*bs,mini_tl2,-1)
                visionnew2 = visionnew2.unsqueeze(1).expand(-1,bs,-1,-1).contiguous().view(bs*bs,mini_tl2,-1)
                      
        

        if self.MODE == 'shrinktofast':
            extended_attention_mask_text = non_pad_mask.squeeze()[:, None, None, :]
            extended_attention_mask_text = (1.0 - extended_attention_mask_text) * -10000.0
            extended_attention_mask_vision = vision_mask.squeeze()[:, None, None, :]
            extended_attention_mask_vision = (1.0 - extended_attention_mask_vision) * -10000.0
            textnew1 = self.encoder2.encoder(text_output,extended_attention_mask_text,head_mask)
            visionnew1 = self.encoder2.encoder(vision_feat,extended_attention_mask_vision,head_mask)
            textnew1 = textnew1[0]
            visionnew1 = visionnew1[0]
            visionnew1 = F.normalize(visionnew1, p=2, dim=2)
            textnew1 = F.normalize(textnew1, p=2, dim=2)
            mini_tl = self.T_length
            visionnew1 = visionnew1[:, :mini_tl]
            textnew1 = textnew1[:, :mini_tl]

            textnew2 = self.encoder3.encoder(text_output,extended_attention_mask_text,head_mask)
            visionnew2 = self.encoder3.encoder(vision_feat,extended_attention_mask_vision,head_mask)
            textnew2 = textnew2[0]
            visionnew2 = visionnew2[0]
            visionnew2 = F.normalize(visionnew2, p=2, dim=2)
            textnew2 = F.normalize(textnew2, p=2, dim=2)
            mini_tl2 = self.S_length
            visionnew2 = visionnew2[:, :mini_tl2]
            textnew2 = textnew2[:, :mini_tl2]

            if istest == False:
                textnew1 = textnew1.unsqueeze(0).expand(bs,-1,-1,-1).contiguous().view(bs*bs,mini_tl,-1)
                visionnew1 = visionnew1.unsqueeze(1).expand(-1,bs,-1,-1).contiguous().view(bs*bs,mini_tl,-1)
                textnew2 = textnew2.unsqueeze(0).expand(bs,-1,-1,-1).contiguous().view(bs*bs,mini_tl2,-1)
                visionnew2 = visionnew2.unsqueeze(1).expand(-1,bs,-1,-1).contiguous().view(bs*bs,mini_tl2,-1)
                      

        if istest == False:
            scores = torch.bmm(textnew1, visionnew1.permute(0, 2, 1)).view(bs, bs, mini_tl, mini_tl)
            scores_t = scores.max(dim=3)[0].sum(2)
            if self.MODE == 'shrinktobase' or self.MODE == 'shrinktofast':
                scores = torch.bmm(textnew2, visionnew2.permute(0, 2, 1)).view(bs, bs, mini_tl2, mini_tl2)
                scores_t2 = scores.max(dim=3)[0].sum(2)
        else:  
            scores = torch.bmm(textnew1, visionnew1.permute(0, 2, 1)).view(bs, mini_tl, mini_tl)
            scores_t = scores.max(dim=2)[0].sum(1)
            if self.MODE == 'shrinktobase' or self.MODE == 'shrinktofast':
                scores = torch.bmm(textnew2, visionnew2.permute(0, 2, 1)).view(bs, mini_tl2, mini_tl2)
                scores_t2 = scores.max(dim=2)[0].sum(1)

        if istest:  
            if self.MODE == 'shrinktobase' or self.MODE == 'shrinktofast':
                return scores_t2
            else:
                return scores_t
        else:
            if self.MODE == 'shrinktobase' or self.MODE == 'shrinktofast':
                return  self.compute_distill(scores_t, scores_t2)
            else:
                return  com_cost(scores_t, self.MARGIN, bs)

def com_cost(scores, margin, bs):
            scores_c = scores.view(bs,bs)
            diagonal = scores_c.diag().view(scores_c.size(0), 1)
            d1 = diagonal.expand_as(scores_c) 
            d2 = diagonal.t().expand_as(scores_c) 
            cost_s = (margin + scores_c - d1).clamp(min=0) 
            cost_im = (margin + scores_c - d2).clamp(min=0)
            eps = 1e-5
            cost_s = cost_s.pow(8).sum(1).add(eps).sqrt().sqrt().sqrt()
            cost_im = cost_im.pow(8).sum(0).add(eps).sqrt().sqrt().sqrt()
            cost_c = cost_s.sum() + cost_im.sum()
            return cost_c


def cosine_similarity(x1, x2, dim=1, eps=1e-5):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)
    
    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip

        self.txt_enc = EncoderText(opt)
 
        self.drop = torch.nn.Dropout(p=0.0)

        self.txt_enc.cuda()
        cudnn.benchmark = True

        if opt.MODE == 'shrinktofast':
            params = list(self.txt_enc.encoder3.parameters())
        elif opt.MODE == 'shrinktobase':
            params = list(self.txt_enc.encoder2.parameters())
        else :
            params = list(self.txt_enc.parameters())

        self.params = params
        self.optimizer = torch.optim.AdamW(params, lr=opt.learning_rate)
        self.txt_enc, self.optimizer = amp.initialize(self.txt_enc, self.optimizer, opt_level= "O1")
        self.txt_enc = torch.nn.DataParallel(self.txt_enc)
        # Loss and Optimizer
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.txt_enc.load_state_dict(state_dict[0], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.txt_enc.eval()

    def forward_emb(self, images, captions,  target_mask, vision_mask, volatile=False, istest = False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images.float(), volatile=volatile)
        captions = torch.LongTensor(captions)
        captions = Variable(captions, volatile=volatile)
       
        # Forward

        n_img = images.size(0)
        n_cap = captions.size(0)

        if istest:
            images = images.unsqueeze(1).expand(n_img,n_cap,images.size(1),images.size(2)).contiguous().view(-1,images.size(1),images.size(2))
            captions = captions.unsqueeze(0).expand(n_img,n_cap,captions.size(1)).contiguous().view(-1,captions.size(1))

        attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        token_type_ids = torch.zeros_like(attention_mask)

        video_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()
        if istest:
            video_non_pad_mask = video_non_pad_mask.unsqueeze(1).expand(n_img,n_cap,images.size(1)).contiguous().view(-1,images.size(1))

        scores = self.txt_enc(captions, token_type_ids, attention_mask,images,video_non_pad_mask, istest)
        return scores


    def train_emb(self, images, captions, target_mask, vision_mask, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
         
        # measure accuracy and record loss
        scores = self.forward_emb(images, captions, target_mask, vision_mask)
        # measure accuracy and record loss

        self.optimizer.zero_grad()
        if scores is not None:
           loss = scores.sum()
        else:
           return
        # compute gradient and do SGD step
        #loss.backward()
        self.logger.update('Loss', loss.item())

        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
           scaled_loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()



