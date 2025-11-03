from openrlhf.trainer import KDTrainer
from typing import Optional, Tuple,List 
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import os
from abc import ABC
import torch

from torch.optim import Optimizer
from tqdm import tqdm
from openrlhf.models import GPTLMLoss, KDLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
import logging
from typing import Optional, Tuple
import torch
from torch.distributed import get_rank, get_world_size, all_gather, broadcast
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
class AdaptiveKLWeightedKLLoss(nn.Module):
    """
    KL Divergence Loss with Adaptive KL-Based Weighting
    """
    def __init__(self, temperature=1.0, batch_norm=False,attention_dillution_coef=1):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.temperature = temperature
        self.epsilon = 1e-4  # 防止除零
        self.attention_dillution_coef = attention_dillution_coef
        self.batch_norm = batch_norm
    
    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, 
                student_label: torch.Tensor, teacher_label: torch.Tensor,gold_idxs: List[int]) -> torch.Tensor:
        """
        Inputs:
        - logits: Student logits, shape [B, L, C]
        - teacher_logits: Teacher logits, shape [B, L, C]
        - *_label: token-level masks, shape [B, L]
        """
        gold_idxs = torch.tensor(gold_idxs, device=logits.device)
        
        teacher_mask = (teacher_label != self.IGNORE_INDEX)  # [B, L]
        student_mask = (student_label != self.IGNORE_INDEX)  # [B, L]
        B = teacher_mask.shape[0]
        assert torch.all(torch.sum(teacher_mask, dim=1) == torch.sum(student_mask, dim=1)), "Mask mismatch"

        teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1) # [B, L, C]
        student_log_probs = F.log_softmax(logits / self.temperature, dim=-1)         # [B, L, C]
        kl_divs = torch.stack([
            F.kl_div(
                student_log_probs[i][student_mask[i]], 
                teacher_log_probs[i][teacher_mask[i]].exp(),
                reduction="batchmean"
            )
            for i in range(B)
        ])
        # mask = gold_idxs == 0
        # local_invaried_tensor = kl_divs[mask].sum() # [1]
        # local_kl_tensor = kl_divs[~mask] # [3]
        local_kl_tensor = kl_divs
        print(f"rank = {get_rank()}")       
        print(f"local_kl_tensor = {local_kl_tensor}")
        print(f"gold idxs = {gold_idxs}")
        if self.batch_norm:
            world_size = dist.get_world_size() 
            gathered_kl_list = [torch.zeros_like(local_kl_tensor) for _ in range(world_size)]
            gathered_gold_indexs = [torch.zeros_like(gold_idxs) for _ in range(world_size)]
            all_gather(gathered_kl_list, local_kl_tensor)
            all_gather(gathered_gold_indexs, gold_idxs)
            all_kl = torch.stack(gathered_kl_list, dim=0) # [8, 4]
            all_gold_indexs = torch.stack(gathered_gold_indexs, dim=0) #[8,4]
            print(f"all_kl.shape = {all_kl.shape}")
            print(f"all_gold_index shape = {all_gold_indexs.shape}")
            # 仅在Rank 0计算权重后广播，确保一致性
            if dist.get_rank() == 0:
                weights = torch.zeros_like(all_gold_indexs,dtype=torch.float32)
                mask = all_gold_indexs == 0
                weights[mask] = self.attention_dillution_coef
                unique_gold_indexs = torch.unique(all_gold_indexs[~mask])
                # unique_gold_indexs = torch.unique(all_gold_indexs)
                pos_mask_list = {}
                pos_kl_list = []
                for unique_gold_index in unique_gold_indexs:
                    pos_mask = all_gold_indexs == unique_gold_index
                    avg_pos_kl = torch.sum(all_kl[pos_mask])/torch.sum(pos_mask)
                    # add frequency
                
                    pos_kl_list.append(avg_pos_kl)
                print(f"uniaue gold indexs = {unique_gold_indexs}")
                print(f"pos_kl_list = {pos_kl_list}")

                pos_kl_tensor = torch.stack(pos_kl_list,dim=0)
                # pos_kl_coefs = pos_kl_tensor*pos_kl_tensor.numel()/(torch.sum(pos_kl_tensor)+self.epsilon)
                pos_kl_coefs =F.softmax(pos_kl_tensor,dim=0)*pos_kl_tensor.numel()
                # pos_kl_coefs =F.softmax(pos_kl_tensor,dim=0)

            
                print(f"pos_kl_coefs = {pos_kl_coefs}")

                for pos_idx,unique_gold_index in enumerate(unique_gold_indexs):
                    pos_mask = all_gold_indexs == unique_gold_index
                    weights[pos_mask] = pos_kl_coefs[pos_idx]/torch.sum(pos_mask)*(all_kl[pos_mask]/max(all_kl[pos_mask]))

            else:
                weights = torch.zeros_like(all_gold_indexs,dtype=torch.float32)
            dist.broadcast(weights, src=0)

            print(f"rank = {get_rank()}, weights = {weights}")
            print(f"rank = {get_rank()}, all_kl = {all_kl}")
            total_loss = torch.sum(weights[get_rank()].detach() * local_kl_tensor)
        else:
            total_loss = torch.sum(local_kl_tensor) + local_invaried_tensor
            # total_loss = torch.sum(torch.clamp((local_kl_tensor.numel() * local_kl_tensor / torch.sum(local_kl_tensor)).detach(),min=0.3,max=self.attention_dillution_coef) *  local_kl_tensor) + local_invaried_tensor * self.attention_dillution_coef
            # total_loss = torch.sum(local_kl_tensor.numel() * local_kl_tensor / torch.sum(local_kl_tensor).detach() *  local_kl_tensor) + local_invaried_tensor * self.attention_dillution_coef
            # total_loss = torch.sum(local_kl_tensor.numel() * local_kl_tensor / torch.sum(local_kl_tensor).detach() *  local_kl_tensor)

        print(f"total_loss = {total_loss}")
        return total_loss * (self.temperature ** 2)


class AdaptiveKLWeightedKLLoss_multi(nn.Module):
    """
    KL Divergence Loss with Adaptive KL-Based Weighting
    """
    def __init__(self, temperature=1.0, batch_norm=False,attention_dillution_coef=1):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.temperature = temperature
        self.epsilon = 1e-4  # 防止除零
        self.attention_dillution_coef = attention_dillution_coef
        self.batch_norm = batch_norm
    
    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, 
                student_label: torch.Tensor, teacher_label: torch.Tensor,gold_idxs: List[int]) -> torch.Tensor:
        """
        Inputs:
        - logits: Student logits, shape [B, L, C]
        - teacher_logits: Teacher logits, shape [B, L, C]
        - *_label: token-level masks, shape [B, L]
        """
        gold_idxs = torch.tensor(gold_idxs, device=logits.device)
        
        teacher_mask = (teacher_label != self.IGNORE_INDEX)  # [B, L]
        student_mask = (student_label != self.IGNORE_INDEX)  # [B, L]
        B = teacher_mask.shape[0]
        assert torch.all(torch.sum(teacher_mask, dim=1) == torch.sum(student_mask, dim=1)), "Mask mismatch"

        teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1) # [B, L, C]
        student_log_probs = F.log_softmax(logits / self.temperature, dim=-1)         # [B, L, C]
        # breakpoint()
        kl_divs = torch.stack([
            F.kl_div(
                student_log_probs[i][student_mask[i]], 
                teacher_log_probs[i][teacher_mask[i]].exp(),
                reduction="batchmean"
            )
            for i in range(B)
        ])
        # mask = gold_idxs == 0
        # local_invaried_tensor = kl_divs[mask].sum() # [1]
        # local_kl_tensor = kl_divs[~mask] # [3]
        local_kl_tensor = kl_divs
        print(f"rank = {get_rank()}")       
        print(f"local_kl_tensor = {local_kl_tensor}")
        print(f"gold idxs = {gold_idxs}")
        print(f"observed pos: {torch.sum(teacher_mask, dim=1)}")
        if self.batch_norm:
            world_size = dist.get_world_size() 
            gathered_kl_list = [torch.zeros_like(local_kl_tensor) for _ in range(world_size)]
            gathered_gold_indexs = [torch.zeros_like(gold_idxs) for _ in range(world_size)]
            all_gather(gathered_kl_list, local_kl_tensor)
            all_gather(gathered_gold_indexs, gold_idxs)
            all_kl = torch.stack(gathered_kl_list, dim=0) # [8, 4]
            all_gold_indexs = torch.stack(gathered_gold_indexs, dim=0) #[8,4,2]
            print(f"all_kl.shape = {all_kl.shape}")
            print(f"all_gold_index shape = {all_gold_indexs.shape}")
            # 仅在Rank 0计算权重后广播，确保一致性
            if dist.get_rank() == 0:
                m,n,_ = all_gold_indexs.size()
                weights = torch.zeros((m,n),dtype=torch.float32,device=all_gold_indexs.device)
                # weights = F.softmax(all_kl.flatten())*all_kl.numel().view(m,n)
                # mask = (all_gold_indexs == torch.tensor([0, 1],device=all_gold_indexs.device)).all(dim=-1)
                # weights[mask] = self.attention_dillution_coef
                # weights[~mask] = all_kl[~mask]/torch.sum(all_kl[~mask])*all_kl[~mask].numel()
                # weights = F.softmax(all_kl.flatten(),dim=0).view(m,n)*all_kl.numel()
                weights = all_kl.flatten()/torch.sum(all_kl.flatten()+self.epsilon)*all_kl.numel()
                


                # mask = all_gold_indexs == 0
                # weights[mask] = self.attention_dillution_coef
                # unique_gold_indexs = torch.unique(all_gold_indexs[~mask])
                # # unique_gold_indexs = torch.unique(all_gold_indexs)
                # pos_mask_list = {}
                # pos_kl_list = []
                # for unique_gold_index in unique_gold_indexs:
                #     pos_mask = all_gold_indexs == unique_gold_index
                #     avg_pos_kl = torch.sum(all_kl[pos_mask])/torch.sum(pos_mask)
                #     # add frequency
                
                #     pos_kl_list.append(avg_pos_kl)
                # print(f"unique gold indexs = {unique_gold_indexs}")
                # print(f"pos_kl_list = {pos_kl_list}")

                # pos_kl_tensor = torch.stack(pos_kl_list,dim=0)
                # # pos_kl_coefs = pos_kl_tensor*pos_kl_tensor.numel()/(torch.sum(pos_kl_tensor)+self.epsilon)
                # pos_kl_coefs =F.softmax(pos_kl_tensor,dim=0)*pos_kl_tensor.numel()
                # # pos_kl_coefs =F.softmax(pos_kl_tensor,dim=0)

            
                # print(f"pos_kl_coefs = {pos_kl_coefs}")

                # for pos_idx,unique_gold_index in enumerate(unique_gold_indexs):
                #     pos_mask = all_gold_indexs == unique_gold_index
                #     weights[pos_mask] = pos_kl_coefs[pos_idx]/torch.sum(pos_mask)*(all_kl[pos_mask]/max(all_kl[pos_mask]))

            else:
                m,n,_ = all_gold_indexs.size()
                weights = torch.zeros((m,n),dtype=torch.float32,device=all_gold_indexs.device)
            dist.broadcast(weights, src=0)

            print(f"rank = {get_rank()}, weights = {weights}")
            print(f"rank = {get_rank()}, all_kl = {all_kl}")
            total_loss = torch.sum(weights[get_rank()].detach() * local_kl_tensor)
        else:
            total_loss = torch.sum(local_kl_tensor) + local_invaried_tensor
            # total_loss = torch.sum(torch.clamp((local_kl_tensor.numel() * local_kl_tensor / torch.sum(local_kl_tensor)).detach(),min=0.3,max=self.attention_dillution_coef) *  local_kl_tensor) + local_invaried_tensor * self.attention_dillution_coef
            # total_loss = torch.sum(local_kl_tensor.numel() * local_kl_tensor / torch.sum(local_kl_tensor).detach() *  local_kl_tensor) + local_invaried_tensor * self.attention_dillution_coef
            # total_loss = torch.sum(local_kl_tensor.numel() * local_kl_tensor / torch.sum(local_kl_tensor).detach() *  local_kl_tensor)

        print(f"total_loss = {total_loss}")
        return total_loss * (self.temperature ** 2)
# class AdaptiveKLWeightedKLLoss(nn.Module):
#     """
#     KL Divergence Loss with Adaptive KL-Based Weighting
#     """
#     def __init__(self, temperature=1.0, batch_norm=False,attention_dillution_coef=1):
#         super().__init__()
#         self.IGNORE_INDEX = -100
#         self.temperature = temperature
#         self.epsilon = 1e-4  # 防止除零
#         self.attention_dillution_coef = attention_dillution_coef
#         self.batch_norm = batch_norm
    
#     def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, 
#                 student_label: torch.Tensor, teacher_label: torch.Tensor,gold_idxs: List[int]) -> torch.Tensor:
#         """
#         Inputs:
#         - logits: Student logits, shape [B, L, C]
#         - teacher_logits: Teacher logits, shape [B, L, C]
#         - *_label: token-level masks, shape [B, L]
#         """
#         gold_idxs = torch.tensor(gold_idxs, device=logits.device)
        
#         teacher_mask = (teacher_label != self.IGNORE_INDEX)  # [B, L]
#         student_mask = (student_label != self.IGNORE_INDEX)  # [B, L]
#         B = teacher_mask.shape[0]
#         assert torch.all(torch.sum(teacher_mask, dim=1) == torch.sum(student_mask, dim=1)), "Mask mismatch"

#         teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1) # [B, L, C]
#         student_log_probs = F.log_softmax(logits / self.temperature, dim=-1)         # [B, L, C]
#         kl_divs = torch.stack([
#             F.kl_div(
#                 student_log_probs[i][student_mask[i]], 
#                 teacher_log_probs[i][teacher_mask[i]].exp(),
#                 reduction="batchmean"
#             )
#             for i in range(B)
#         ])
#         # mask = gold_idxs == 0
#         # local_invaried_tensor = kl_divs[mask].sum() # [1]
#         # local_kl_tensor = kl_divs[~mask] # [3]
#         local_kl_tensor = kl_divs
#         print(f"rank = {get_rank()}")       
#         print(f"local_kl_tensor = {local_kl_tensor}")
#         print(f"gold idxs = {gold_idxs}")
#         if self.batch_norm:
#             world_size = dist.get_world_size() 
#             gathered_kl_list = [torch.zeros_like(local_kl_tensor) for _ in range(world_size)]
#             gathered_gold_indexs = [torch.zeros_like(gold_idxs) for _ in range(world_size)]
#             all_gather(gathered_kl_list, local_kl_tensor)
#             all_gather(gathered_gold_indexs, gold_idxs)
#             all_kl = torch.stack(gathered_kl_list, dim=0) # [8, 4]
#             all_gold_indexs = torch.stack(gathered_gold_indexs, dim=0) #[8,4]
#             print(f"all_kl.shape = {all_kl.shape}")
#             print(f"all_gold_index shape = {all_gold_indexs.shape}")
#             # 仅在Rank 0计算权重后广播，确保一致性
#             if dist.get_rank() == 0:
#                 mask = all_gold_indexs == 0
#                 weights = torch.zeros_like(all_gold_indexs,dtype=torch.float32)
#                 weights[mask] = self.attention_dillution_coef
#                 print(f"mask = {mask}")
#                 print(f"~mask = {~mask}")
#                 weights[~mask] = all_kl[~mask]*torch.sum(~mask)/(torch.sum(all_kl[~mask]) + self.epsilon)
#             else:
#                 weights = torch.zeros_like(all_gold_indexs,dtype=torch.float32)
#             dist.broadcast(weights, src=0)

#             print(f"rank = {get_rank()}, weights = {weights}")
#             print(f"rank = {get_rank()}, all_kl = {all_kl}")
#             total_loss = torch.sum(weights[get_rank()].detach() * local_kl_tensor)
#         else:
#             total_loss = torch.sum(local_kl_tensor) + local_invaried_tensor
#             # total_loss = torch.sum(torch.clamp((local_kl_tensor.numel() * local_kl_tensor / torch.sum(local_kl_tensor)).detach(),min=0.3,max=self.attention_dillution_coef) *  local_kl_tensor) + local_invaried_tensor * self.attention_dillution_coef
#             # total_loss = torch.sum(local_kl_tensor.numel() * local_kl_tensor / torch.sum(local_kl_tensor).detach() *  local_kl_tensor) + local_invaried_tensor * self.attention_dillution_coef
#             # total_loss = torch.sum(local_kl_tensor.numel() * local_kl_tensor / torch.sum(local_kl_tensor).detach() *  local_kl_tensor)

#         print(f"total_loss = {total_loss}")
#         return total_loss * (self.temperature ** 2)
class MY_KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """
    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.temperature = 1.0
        self.chunk_size = 64
        self.decay_factor = 0.99
    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, 
            student_label: torch.Tensor, teacher_label: torch.Tensor) -> torch.Tensor:
            # 计算教师和学生的有效 token mask
            teacher_mask = (teacher_label != self.IGNORE_INDEX)  # [B, L]
            student_mask = (student_label != self.IGNORE_INDEX)  # [B, L]
            # 确保 mask 位置一致
            assert torch.sum(teacher_mask) == torch.sum(student_mask), "Teacher and student masks must have the same number of valid tokens"
            B = teacher_mask.shape[0]
            L = teacher_mask.shape[1]
            teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)  # 教师 log_softmax
            student_log_probs = F.log_softmax(logits / self.temperature, dim=-1, dtype=torch.float32)  # [B, L, V]

            position_weights = torch.tensor([3*self.decay_factor ** i for i in range(L)], dtype=torch.float32, device=teacher_logits.device)         
            # 对每个样本的有效位置计算 KL loss
            kl_divs = []
            for i in range(B):
                kl_loss = F.kl_div(student_log_probs[i][student_mask[i]], teacher_log_probs[i][teacher_mask[i]].exp(),reduction="none") # seq*V
                # breakpoint()

                kl_loss = torch.mean(torch.sum(kl_loss,dim=-1)*position_weights[:torch.sum(student_mask[i])])

                kl_divs.append(kl_loss)
            kl_divs = torch.stack(kl_divs)
            # 计算最终的 KL loss
            return kl_divs.mean()
  
# class MY_KDLoss(nn.Module):
#     """
#     Language Model Knowledge Distillation Loss
#     """
#     def __init__(self):
#         super().__init__()
#         self.IGNORE_INDEX = -100
#         self.temperature = 1.0
#     def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, 
#             student_label: torch.Tensor, teacher_label: torch.Tensor) -> torch.Tensor:
#             # 计算教师和学生的有效 token mask
#             teacher_mask = (teacher_label != self.IGNORE_INDEX)  # [B, L]
#             student_mask = (student_label != self.IGNORE_INDEX)  # [B, L]
#             # 确保 mask 位置一致
#             B = teacher_mask.shape[0]
#             assert torch.sum(teacher_mask) == torch.sum(student_mask), "Teacher and student masks must have the same number of valid tokens"

#             teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)  # 教师 log_softmax
#             student_log_probs = F.log_softmax(logits / self.temperature, dim=-1, dtype=torch.float32)  # 学生 log_softmax
      
#             kl_divs = torch.stack([
#                 F.kl_div(
#                     student_log_probs[i][student_mask[i]], 
#                     teacher_log_probs[i][teacher_mask[i]].exp(),
#                     reduction="batchmean"
#                 )
#                 for i in range(B)
#             ])
#             kl_loss = kl_divs.mean()
#             return kl_loss * (self.temperature ** 2)
class MY_rankLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """
    def __init__(self,top_k=30, margin=0.5):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.temperature = 2.0
        self.top_k = top_k
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, 
            student_label: torch.Tensor, teacher_label: torch.Tensor) -> torch.Tensor:
            # 计算教师和学生的有效 token mask
            teacher_mask = (teacher_label != self.IGNORE_INDEX).unsqueeze(-1)  # [B, L, 1]
            student_mask = (student_label != self.IGNORE_INDEX).unsqueeze(-1)  # [B, L, 1]
            # 确保 mask 位置一致
            assert torch.sum(teacher_mask) == torch.sum(student_mask), "Teacher and student masks must have the same number of valid tokens"
            teacher_indices = teacher_mask.squeeze(-1).nonzero(as_tuple=True)
            student_indices = student_mask.squeeze(-1).nonzero(as_tuple=True)
            # 提取有效 token 的概率
            teacher_logits = teacher_logits[teacher_indices]  # [有效Token数, C]
            student_logits = logits[student_indices]  # [有效Token数, C]
            topk_values, topk_indices = torch.topk(teacher_logits, self.top_k, dim=-1)  # 获取 top-k token
            student_topk_logits = torch.gather(student_logits, dim=-1, index=topk_indices)
            # 生成pairs
            i_indices, j_indices = torch.triu_indices(self.top_k, self.top_k, offset=1)
            # 计算pairwise差异
            teacher_i = topk_values[:, i_indices]
            teacher_j = topk_values[:, j_indices]
            student_i = student_topk_logits[:, i_indices] 
            student_j = student_topk_logits[:, j_indices]
            
            # 生成ranking标签
            ranking_labels = (teacher_i > teacher_j).float() * 2 - 1
            
            # 添加权重
            rank_diff = torch.abs(i_indices - j_indices).float()
            weights = 1.0 / (rank_diff + 1)
            weights = weights.to(student_i.device)
            
            # 计算loss
            ranking_loss = (self.ranking_loss(student_i, student_j, ranking_labels) * weights).mean()
            
            return ranking_loss
class MY_topkLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """
    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.temperature = 1.0
        self.top_k = 20
    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, 
            student_label: torch.Tensor, teacher_label: torch.Tensor) -> torch.Tensor:
            # 计算教师和学生的有效 token mask
            teacher_mask = (teacher_label != self.IGNORE_INDEX)  # [B, L]
            student_mask = (student_label != self.IGNORE_INDEX)  # [B, L]
            # 确保 mask 位置一致
            assert torch.sum(teacher_mask) == torch.sum(student_mask), "Teacher and student masks must have the same number of valid tokens"
            teacher_indices = teacher_mask.squeeze(-1).nonzero(as_tuple=True)
            student_indices = student_mask.squeeze(-1).nonzero(as_tuple=True)
            # 提取有效 token 的概率
            teacher_logits = teacher_logits[teacher_indices]  # [有效Token数, C]
            student_logits = logits[student_indices]  # [有效Token数, C]
            topk_values, topk_indices = torch.topk(teacher_logits, self.top_k, dim=-1)  # 获取 top-k token
            student_topk_logits = torch.gather(student_logits, dim=-1, index=topk_indices)


            teacher_probs = F.softmax(topk_values / self.temperature, dim=-1)
            student_log_probs = F.log_softmax(student_topk_logits / self.temperature, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            return kl_loss * (self.temperature ** 2)
class MY_GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention

        if self.ring_attn_group is not None:
            logger.info(f" gptloss is entered: the first choice of ring_attn_group.")
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            # breakpoint()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            all_logits = []
            all_labels = []
            batch_size = shift_labels.shape[0]
            shift_mask = (shift_labels != self.IGNORE_INDEX).int().unsqueeze(-1)
            for i in range(batch_size):
                shift_nonzero_indices = shift_mask[i].nonzero()[:,0]
                shift_nonzero_logits = shift_logits[i,shift_nonzero_indices,:]
                shift_nonzero_labels = shift_labels[i,shift_nonzero_indices]
                all_logits.append(shift_nonzero_logits)
                all_labels.append(shift_nonzero_labels)

            # 将所有batch的数据拼接起来
            concatenated_logits = torch.cat(all_logits, dim=0)  # shape: [total_valid_tokens, vocab_size]
            concatenated_labels = torch.cat(all_labels, dim=0)   # shape: [total_valid_tokens]

            # 统一计算loss
            loss = self.loss(concatenated_logits, concatenated_labels)

            return loss
