# TODO

Test different models and their performance.

```bash
# Qwen2.5-3B 20250428-233442
/data1/linkdom/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
# Qwen2.5-3B-Instruct 20250428-224513
/data1/linkdom/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1

# Qwen2.5-1.5B 20250428-233602
/data1/linkdom/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
# Qwen2.5-1.5B-Instruct 20250428-230824
/data1/linkdom/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

# Qwen2.5-Math-1.5B 20250429-160531
/data1/linkdom/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2
# Qwen2.5-Math-1.5B-Instruct 20250429-160713
/data1/linkdom/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35
```

- `git diff config.yaml`

```diff
diff --git a/config.yaml b/config.yaml
index 88a16ba..1ee6708 100644
--- a/config.yaml
+++ b/config.yaml
@@ -19,7 +19,7 @@ training:
   betas: [0.9, 0.999]
   ckpt_dir: "ckpt"
   log_dir: "logs"
-  skip_unfinished_episodes: false
+  skip_unfinished_episodes: true
   ckpt_save_interval: 100
   eval_interval: 10
   memory_efficient_adamw: false
```

- `diff -u grpo.py rgrpo.py`

```diff
--- grpo.py     2025-04-27 11:53:58.965893871 +0800
+++ rgrpo.py    2025-05-04 16:40:16.357436497 +0800
@@ -123,9 +123,8 @@
     for group in groups.values():
         group_rewards = [item.reward for item in group]
         mean_reward = np.mean(group_rewards)
-        std_reward = np.std(group_rewards)
         for episode in group:
-            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
+            normalized_reward = episode.reward - mean_reward
             episode = dataclasses.replace(episode, reward=normalized_reward)
             output.append(episode)
     return output
```

- `diff -u grpo.py lgrpo.py`

```diff
--- grpo.py     2025-04-27 11:53:58.965893871 +0800
+++ lgrpo.py    2025-05-04 16:35:54.737845087 +0800
@@ -168,6 +168,9 @@
             for episode in batch_episodes
         ]
         batch_max_length = max(batch_lengths)
+        batch_max_gen_length = max(
+            len(episode.generated_token_ids) for episode in batch_episodes
+        )
         batch_token_ids = [
             episode.prefix_token_ids
             + episode.generated_token_ids
@@ -206,7 +209,7 @@
 
         obj = log_probs * batch_advantages[:, None]
         # per-token objective
-        obj = (obj * target_masks).sum() / num_target_tokens
+        obj = (obj * target_masks).sum() / batch_max_gen_length
         loss = -obj
         loss.backward()
```

- `diff -u grpo.py sgrpo.py`

```diff 
--- grpo.py     2025-04-27 11:53:58.965893871 +0800
+++ sgrpo.py    2025-05-04 15:44:09.538893672 +0800
@@ -204,7 +204,9 @@
             token_entropy = compute_entropy(logits)
             entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens
 
-        obj = log_probs * batch_advantages[:, None]
+        # 先设置一个固定的值来试一下, 看后面要不要改成 adaptive 的
+        ent_coef = 0.1
+        obj = log_probs * (batch_advantages[:, None] + ent_coef * token_entropy)
         # per-token objective
         obj = (obj * target_masks).sum() / num_target_tokens
         loss = -obj
```