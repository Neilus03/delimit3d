from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[2] / 'artifacts/scannet_feature_selectivity_table'
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Serif', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(8.5, 4.65), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
caption = ('Table 1. Feature selectivity on ScanNet validation. We compare public LitePT with the same encoder after 256 Delimit3D adaptation updates, using matched inputs and sampling. Evaluation covers 312 scans, 10,323 eligible instances, and 71,694 ordered same-category instance pairs. Separation margins measure the difference between mean query-to-target and query-to-other-instance cosine similarity; the hardest-instance variant uses the most similar other instance of the same category. Retrieval AP ranks scene points by query-feature similarity. Leakage is the fraction of non-target points among the top-ranked points, with the retrieval budget equal to the target instance size. Results are averaged equally across scans. AP and leakage are percentages; their changes are percentage points.')
fig.text(.075, .925, textwrap.fill(caption, 107), fontsize=9.5, va='top', linespacing=1.4)
xs = [.075, .66, .815, .925]
header_y = .455
for x, txt, align in zip(xs, ['Metric', 'LitePT', '+ Delimit3D', 'Delta'], ['left','right','right','right']):
    fig.text(x, header_y, txt, fontsize=10, weight='bold', ha=align, va='center')
rows = [
 ('Same-category pair margin ↑', '0.233', '0.303', '+0.071'),
 ('Hardest same-category instance margin ↑', '0.087', '0.126', '+0.039'),
 ('Instance retrieval AP (%) ↑', '49.17', '56.77', '+7.60'),
 ('Non-target leakage (%) ↓', '53.00', '46.25', '−6.75'),
]
for i, row in enumerate(rows):
    y = .375 - i*.071
    for j, (x, txt) in enumerate(zip(xs, row)):
        fig.text(x, y, txt, fontsize=10, ha='left' if j==0 else 'right', va='center', weight='bold' if j==2 else 'normal')
for y,lw in [(.497,1.1),(.415,.65),(.112,1.1)]:
    ax.plot([.075,.925],[y,y],color='black',lw=lw,transform=fig.transFigure,clip_on=False)
fig.savefig(out/'scannet_feature_selectivity_table.pdf',metadata={'Title':'Feature selectivity on ScanNet validation'})
fig.savefig(out/'scannet_feature_selectivity_table.png',dpi=160)
print(out/'scannet_feature_selectivity_table.pdf')
