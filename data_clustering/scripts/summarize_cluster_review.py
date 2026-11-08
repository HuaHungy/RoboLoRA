import json, itertools, shutil, re
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('reports/parameter_review')
results=json.loads((out/'combined_results.json').read_text())
rows=[json.loads(s) for s in Path('runs/lerobot_ee_all/windows/index.jsonl').read_text().splitlines()]
val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
seed_records=[];seeds_by_model={}
for trial in ['baseline128','latent64']:
 baseline=next(r for r in results if r['trial']==trial and r['epoch']==30 and r['k']==32)
 group=[{**baseline,'training_seed':42}]
 labels=[np.load(Path(baseline['checkpoint']).parent/'labels_k32.npy')[val]]
 for seed in [7,123]:
  f=Path('runs')/f'parameter_review_{trial}_seed{seed}'/'results.json'
  r=json.loads(f.read_text())['results'][0];group.append(r)
  labels.append(np.load(Path(r['checkpoint']).parent/'labels_k32.npy')[val])
 aris=[float(adjusted_rand_score(a,b)) for a,b in itertools.combinations(labels,2)]
 seeds_by_model[trial]={'records':group,'mean_silhouette':float(np.mean([r['validation_silhouette'] for r in group])),
  'mean_validation_action_mse':float(np.mean([r['validation_action_mse'] for r in group])),
  'mean_kmeans_seed0_ari':float(np.mean([r['seed0_validation_ari'] for r in group])),
  'between_training_seed_ari':aris,'mean_between_training_seed_ari':float(np.mean(aris))}
 seed_records+=group
(out/'training_seed_comparison.json').write_text(json.dumps(seeds_by_model,indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in seeds_by_model.items()},indent=2))

fig,axes=plt.subplots(1,3,figsize=(16,4.7))
for trial,color,label in [('baseline128','#3572b0','128D, state weight 1'),('latent64','#2a9d8f','64D, state weight 1'),('state025','#e9a23b','128D, state weight 0.25')]:
 rs=sorted([r for r in results if r['trial']==trial and r['k']==32],key=lambda x:x['epoch'])
 axes[0].plot([r['epoch'] for r in rs],[r['validation_silhouette'] for r in rs],'-o',label=label,color=color)
 axes[1].plot([r['epoch'] for r in rs],[r['validation_action_mse'] for r in rs],'-o',label=label,color=color)
axes[0].set(xlabel='Training epochs',ylabel='Validation silhouette',title='K=32: separation');axes[0].legend(fontsize=8)
axes[1].set(xlabel='Training epochs',ylabel='Validation action MSE (log)',yscale='log',title='Reconstruction continues to improve')
for i,(trial,v) in enumerate(seeds_by_model.items()):
 axes[2].scatter(np.full(3,i),[r['validation_silhouette'] for r in v['records']],s=55,label=trial)
 axes[2].plot([i-.2,i+.2],[v['mean_silhouette']]*2,'k-',lw=2)
axes[2].set(xticks=[0,1],xticklabels=['128D / 30 epochs','64D / 30 epochs'],ylabel='Validation silhouette',title='Three network training seeds')
fig.tight_layout();fig.savefig(out/'parameter_comparison.png',dpi=170);plt.close(fig)

selected=next(r for r in results if r['trial']=='latent64' and r['epoch']==30 and r['k']==32)
original=next(r for r in results if r['trial']=='original_best29')
oldreview=json.loads(Path('reports/k32_visual_review/review_metrics.json').read_text())
newreview=json.loads(Path('reports/k32_latent64_review/review_metrics.json').read_text())
a=np.load('runs/lerobot_ee_all/windows/action.npy',mmap_mode='r');d=a[:,:,14:16]
cycles=int((((np.maximum(d,0).sum(1)>.2)&(np.maximum(-d,0).sum(1)>.2)).any(1)).sum())
counts=Counter(r['dataset_id'] for r in rows)
selection={'recommended_parameters':{'window_length':50,'stride':25,'latent_dim':64,'hidden_dim':256,'epochs':30,'batch_size':256,'lr':.001,'action_loss_weight':1.,'state_loss_weight':1.,'k':32,'n_init':10,'seed':42,'visual_weight':0.},'scope':'best balanced candidate in this bounded review, not proven optimal or semantically validated','grid_candidates':36,'additional_training_seed_runs':4,'review_clips_per_model':128,'selected':selected,'original':original,'training_seed_comparison':seeds_by_model,'gripper_open_and_close_same_window':cycles,'largest_task_window_count':counts.most_common(1)[0],'smallest_task_window_count':counts.most_common()[-1]}
(out/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))

# Keep original logs as traceable evidence, including the interrupted first long batch.
logs=out/'experiment_logs';logs.mkdir(exist_ok=True)
for p in Path('/tmp').glob('parameter_review*.log'):shutil.copy2(p,logs/p.name)

lines=['# 动作聚类复核与参数选择（2026-11-04）','',
'## 结论','',
'建议将 **窗口50、步长25、latent_dim=64、hidden_dim=256、训练30轮、K=32、学习率0.001、batch_size=256、动作/状态损失权重1:1、K-means n_init=10、seed=42** 作为当前较均衡的运动聚类基线。视觉分支保持关闭。该结论仅适用于本次验证集和搜索范围；抽样仍有交互效果混合，不能称为已经验证的原子技能码本。','',
'已导出可用结果至 `runs/lerobot_ee_reviewed64/`；表示为18325×64，32个簇全部有样本。新模型和正式导出的表示、标签与参数实验逐项核对一致。原始30轮结果完整保留。不同运行的簇ID不对应。','',
'## 检查范围','',
'- 原方案K=32：每簇2个不同轨迹的训练代表、2个随机验证片段，共128个；看过全部首/中/尾头部相机画面，并对原簇0、2、6、16、21、23查看11个时间采样点、三路相机。','- 候选64维K=32：同样128个片段，全部首/中/尾画面复核；全部三相机GIF已导出供进一步查看。','- 动态数值覆盖所有18,325个窗口；图像复核是抽样，不代表逐个窗口的人工语义评测。','- 每个GIF每5帧采样一次，显示原始帧索引；名义帧率仅控制播放，不代表精确物理时间。首尾图像取原始HDF5的真实终点。','',
'## 视觉发现','',
'原簇4、31的张开夹爪接近、簇3/7的闭合及抬升、簇22/26的双手闭合等有较一致的运动样式。但原簇0混入盖章、操作电脑盖、持瓶和搬运；原簇2混入小运动保持、摇瓶及交接；原簇6/18/23经常覆盖撤回和另一手接近等多个阶段。','',
'候选64维仍保留上述限制：新簇28同时包含订书机按压附近的运动与持瓶搬运，新簇4包含倾倒/微波炉相关的小运动保持，新簇29的净夹爪事件多数占比只有约29%，仍是需要重点检查的混合簇。相比原结果，视觉抽样没有证明交互语义已经明显改善。','',
'原方案32簇中，有17簇的主运动臂启发式多数占比≥85%，25簇的夹爪净变化模式多数占比≥85%，两者同时达到的为14簇。这只能说明部分运动属性集中，不能当作14个已认证技能。','',
'“hold”字段表示夹爪净变化小于阈值，不保证整个窗口夹爪始终不动。全数据有41个窗口在同一只手上同时累计张开和闭合超过0.2；未据此删掉任何数据。','',
'## 参数比较','',
'固定原有任务内80/20轨迹划分、归一化和窗口，进行了36个组合：128维/权重1的3、10、20、30、50、60轮；64维/权重1的10、20、30轮；128维/状态权重0.25的10、20、30轮；各自K=16/32/48。另比较原来第29轮最佳模型。','',
'表中ARI是在同一网络表示下，将K-means种子换成0、1、2后与正式seed42分配的一致性均值；越接近1越一致。动作MSE只衡量标准化动态的离线重建，不是执行成功率。','',
'| 参数 | 验证动作MSE↓ | 验证轮廓系数↑ | K-means初始化ARI均值↑ |','|---|---:|---:|---:|']
for trial,epoch,label in [('original_best29',29,'原128维，30轮中选29轮'),('baseline128',10,'128维，10轮'),('state025',10,'128维，10轮，状态权重0.25'),('latent64',30,'推荐64维，30轮')]:
 r=next(r for r in results if r['trial']==trial and r['epoch']==epoch and r['k']==32)
 lines.append(f"| {label} | {r['validation_action_mse']:.4f} | {r['validation_silhouette']:.4f} | {r['validation_ari_mean']:.4f} |")
lines+=['',
'64维在seed42上具有较好的初始化稳定性，动作重建误差较原方案高约10%，但相较10轮方案仍小很多。位置分组解释率从0.423提高到0.455，旋转从0.322提高到0.348。解释率计算方法：用训练集中每簇的数值特征均值预测验证窗口，计算相对训练全局均值基线的1−SSE/SST；它是运动一致性代理指标，不是技能准确率。','',
'128维10轮是值得保留的备选：几何分离度和旋转分组解释率较好，但动作重建误差约0.083，比推荐方案约0.0347高。状态权重0.25的10轮虽然轮廓系数较高，但多初始化ARI较低，因此未只按轮廓系数选它。','',
'## 网络训练种子复核','',
'在固定数据划分下，用网络训练种子42、7、123分别验证128维和64维的30轮结果。这里都使用最后第30轮，避免拿第29轮与第30轮作种子均值比较。','',
'| 维度 | 轮廓系数三种子均值↑ | 动作MSE均值↓ | K-means种子0相对42的ARI均值↑ | 网络训练种子间两两ARI均值↑ |','|---|---:|---:|---:|---:|']
for name,v in seeds_by_model.items():lines.append(f"| {name} | {v['mean_silhouette']:.4f} | {v['mean_validation_action_mse']:.4f} | {v['mean_kmeans_seed0_ari']:.4f} | {v['mean_between_training_seed_ari']:.4f} |")
lines+=['',
'64维的几何分离度改善在三个网络种子上都出现；运动解释率并非每个种子、每个维度都改善，因此只称为更均衡的候选。尚无独立测试任务和下游专家成功率，不能宣称统计显著或全局最优。','',
'## 是否需要增加视觉/状态变化/时间一致性','',
'需要，但应围绕实际问题分步验证。首先优先处理窗口相位和动作边界：同一闭合行为因位于窗口早中晚而进入不同簇，相邻50%重叠窗口仍频繁换簇。推荐64维的验证相邻窗口换簇比例约87%，这是诊断值，不是错误率；真实动作转换也会导致换簇。应设计保留真实边界的时间一致性或相位对齐，不能简单强制所有相邻窗口同类。','',
'其次为闭合持物但效果不同的混合簇增加物体相对位移、接触/抬升等效果信息，或以夹爪附近的起止视觉变化作为辅助。仅加入起点全局图像可能让外观和背景主导，不能保证解决交互差异。当前状态分支只是重建已输入的末端首尾变化，尚无物体效果监督。','',
'保留长度50和步长25是为了与已有动作块方案对齐，本次没有比较其他长度；不能声称该窗口长度已经最优。长任务产生更多窗口：最多任务1153个，最少97个，当前未做任务窗口数均衡，这是后续另一项消融因素。','',
'## 产物和复现','',
'- 原方案图库：`reports/k32_visual_review/index.html`；逐簇解释：`visual_observations.json`。','- 候选图库：`reports/k32_latent64_review/index.html`；同样包含32簇、128个片段。','- 参数曲线：`parameter_comparison.png`；完整组合：`combined_results.json`；种子比较：`training_seed_comparison.json`。','- 推荐模型与聚类：`runs/lerobot_ee_reviewed64/`。其中`windows`为指向原始run窗口目录的符号链接，避免重复缓存；移动或清理原run时需要同时保留/复制窗口目录。','- 默认入口仍使用原配置；新增`configs/lerobot_ee_reviewed64.json`可完整重跑推荐参数，写入尚未创建的`runs/lerobot_ee_reviewed64_rerun/`。','',
'```bash','bash /home/huahungy/RoboLoRAHub/data_clustering/run_lerobot_ee.sh /home/huahungy/RoboLoRAHub/data_clustering/configs/lerobot_ee_reviewed64.json','```','',
'已有推荐结果无需再次训练。首次长批次参数实验在50轮评估中中断，未完成候选未计入36组；50/60轮已分批完整重跑。日志保存在`experiment_logs/`，原模型和原聚类结果未覆盖。','',
'## 全部36组指标','',
'| 变体 | 轮数 | K | 动作MSE | 轮廓系数 | 位置解释率 | 旋转解释率 | 夹爪变化解释率 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in results:
 if r['trial']=='original_best29':continue
 lines.append(f"| {r['trial']} | {r['epoch']} | {r['k']} | {r['validation_action_mse']:.4f} | {r['validation_silhouette']:.4f} | {r['feature_r2']['position']:.3f} | {r['feature_r2']['rotation']:.3f} | {r['feature_r2']['gripper_delta']:.3f} |")
(out/'report.md').write_text('\n'.join(lines)+'\n')
# Review artifacts: verify complete GIF sequences, distinct episodes and source split.
from PIL import Image
for gallery in ['k32_visual_review','k32_latent64_review']:
 p=Path('reports')/gallery; items=json.loads((p/'selected_windows.json').read_text())
 assert len(items)==128
 for c in range(32):
  samples=[x for x in items if x['cluster']==c]
  assert len({x['episode_id'] for x in samples})==4
  assert Counter(x['kind'] for x in samples)=={'representative_train':2,'random_validation':2}
 for x in items:
  assert x['split']==('train' if x['kind']=='representative_train' else 'validation')
  with Image.open(p/x['gif']) as im:assert im.n_frames==11
 print(gallery,'128 clips verified')
print('Report generated:',out/'report.md')
