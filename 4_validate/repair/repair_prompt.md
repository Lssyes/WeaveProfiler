任务：修补 {{PROJECT}} 的插桩，使其通过 {{STAGE}} 判定。

【已勘明的事实，直接用，不要重新猜】
{{FACTS}}

【本次要修的缺口】
{{GAPS}}

【自验收】
{{LOOP}}

【全局约束】
- 照参考实现改，不从零发明。措辞是"照 refs/code/<语言>/<类>.py 改"，不是"实现一个 XXX"。
- device 打点写进 kernel 主体，不另起文件。
- host 侧一切进 <项目>/profile/{abi,scripts,harness}/。大产物进 <项目>/.work/，不入库。
- 每条缺口自带禁止项，务必遵守；违反等同 FAIL。
- 只用 GPU {{GPU_ID}}（设 CUDA_VISIBLE_DEVICES={{GPU_ID}}），禁止用其他卡。

【回复格式】
完成后仅回复一行：DONE validate=<PASS/FAIL> correctness=<true/false> events=<数>
超过 3 轮仍 FAIL：回复 BLOCKED 加最后一次 FAIL 原文，不要继续改。
