import matplotlib.pyplot as plt
import numpy as np
import re

# Read the data
data = []
with open('output.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Parse: rank 0 backend:gloo, data_size:262144, world_size:2, time:0.002671962487511337
        match = re.match(r'rank \d+ backend:(\w+), data_size:(\d+), world_size:(\d+), time:([\d.]+)', line)
        if match:
            backend, data_size, world_size, time = match.groups()
            data.append({
                'backend': backend,
                'data_size': int(data_size),
                'world_size': int(world_size),
                'time': float(time)
            })

# Map data_size to labels
data_size_map = {
    262144: '1MB',
    2621440: '10MB',
    26214400: '100MB',
    268435456: '1GB',
    536870912: '2GB'
}

data_size_order = [262144, 2621440, 26214400, 268435456, 536870912]
world_size_order = [2, 3, 4]

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ===== Plot 1: Data Size vs Time (for different world sizes) =====
colors_gloo = ['#e74c3c', '#c0392b', '#a93226']  # Red shades for gloo
colors_nccl = ['#3498db', '#2980b9', '#1f618d']  # Blue shades for nccl
markers = ['o', 's', '^']

for idx, world_size in enumerate(world_size_order):
    # Gloo data
    gloo_times = []
    for data_size in data_size_order:
        matching = [d for d in data if d['backend'] == 'gloo' 
                   and d['world_size'] == world_size 
                   and d['data_size'] == data_size]
        if matching:
            gloo_times.append(matching[0]['time'])
        else:
            gloo_times.append(None)
    
    ax1.plot(range(len(data_size_order)), gloo_times, 
            marker=markers[idx], linestyle='-', linewidth=2, markersize=8,
            color=colors_gloo[idx], label=f'gloo (ws={world_size})')
    
    # NCCL data
    nccl_times = []
    for data_size in data_size_order:
        matching = [d for d in data if d['backend'] == 'nccl' 
                   and d['world_size'] == world_size 
                   and d['data_size'] == data_size]
        if matching:
            nccl_times.append(matching[0]['time'])
        else:
            nccl_times.append(None)
    
    ax1.plot(range(len(data_size_order)), nccl_times, 
            marker=markers[idx], linestyle='--', linewidth=2, markersize=8,
            color=colors_nccl[idx], label=f'nccl (ws={world_size})')

ax1.set_xticks(range(len(data_size_order)))
ax1.set_xticklabels([data_size_map[size] for size in data_size_order])
ax1.set_xlabel('Data Size', fontsize=13, fontweight='bold')
ax1.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
ax1.set_title('Time vs Data Size', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=10, ncol=2)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_yscale('log')

# ===== Plot 2: World Size vs Time (for different data sizes) =====
colors_gloo_2 = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#d4ac0d']
colors_nccl_2 = ['#3498db', '#1abc9c', '#16a085', '#27ae60', '#229954']
markers_2 = ['o', 's', '^', 'd', 'v']

for idx, data_size in enumerate(data_size_order):
    # Gloo data
    gloo_times = []
    for world_size in world_size_order:
        matching = [d for d in data if d['backend'] == 'gloo' 
                   and d['world_size'] == world_size 
                   and d['data_size'] == data_size]
        if matching:
            gloo_times.append(matching[0]['time'])
        else:
            gloo_times.append(None)
    
    ax2.plot(world_size_order, gloo_times, 
            marker=markers_2[idx], linestyle='-', linewidth=2, markersize=8,
            color=colors_gloo_2[idx], label=f'gloo ({data_size_map[data_size]})')
    
    # NCCL data
    nccl_times = []
    for world_size in world_size_order:
        matching = [d for d in data if d['backend'] == 'nccl' 
                   and d['world_size'] == world_size 
                   and d['data_size'] == data_size]
        if matching:
            nccl_times.append(matching[0]['time'])
        else:
            nccl_times.append(None)
    
    ax2.plot(world_size_order, nccl_times, 
            marker=markers_2[idx], linestyle='--', linewidth=2, markersize=8,
            color=colors_nccl_2[idx], label=f'nccl ({data_size_map[data_size]})')

ax2.set_xticks(world_size_order)
ax2.set_xlabel('World Size', fontsize=13, fontweight='bold')
ax2.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
ax2.set_title('Time vs World Size', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper left', fontsize=10, ncol=2)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('communication_benchmark.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'communication_benchmark.png'")
plt.show()

