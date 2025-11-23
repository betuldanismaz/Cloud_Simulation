import gymnasium 
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from cloud_env import CloudEnv

# 1. Ortamı ve Modeli Yükle
env = CloudEnv()
# Önceki eğitimi sıfırdan yapmak yerine, train.py'yi tekrar çalıştırıp
# yeni model oluşunca bunu çalıştıracağız.
# Şimdilik model yoksa hata vermemesi için önce train.py'yi tekrar çalıştır.
try:
    model = DQN.load("dqn_cloud_autoscale")
except:
    print("Önce train.py dosyasını çalıştırıp modeli eğitmelisin!")
    exit()


obs, _ = env.reset()
done = False

# Verileri saklamak için listeler
cpu_history = []
vm_history = []
request_history = []
steps = []

print("Simülasyon ve Grafik Çizimi Başlıyor...")

# 100 Adımlık Test (1 Günlük Simülasyon gibi)
for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    # Verileri kaydet
    steps.append(step)
    cpu_history.append(obs[0])      # CPU
    vm_history.append(obs[1])       # VM Sayısı
    request_history.append(info['request_count']) # Trafik

# --- GRAFİK ÇİZİMİ ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Sol Eksen: CPU Kullanımı
ax1.set_xlabel('Zaman (Adım)')
ax1.set_ylabel('CPU Kullanımı (%)', color='tab:red')
ax1.plot(steps, cpu_history, color='tab:red', label='CPU Load')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='SLA Sınırı (%80)')

# Sağ Eksen: VM Sayısı ve Trafik
ax2 = ax1.twinx()  
ax2.set_ylabel('VM Sayısı', color='tab:blue') 
ax2.step(steps, vm_history, color='tab:blue', label='VM Sayısı (Action)', where='post')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('Yapay Zeka Otonom Ölçekleme Performansı')
fig.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()