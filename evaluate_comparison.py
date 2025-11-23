import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from cloud_env import CloudEnv

# --- AYARLAR ---
TEST_STEPS = 100  # Simülasyon süresi
SEED = 42         # İki yöntemin de AYNI trafiği görmesi için sabit tohum

def run_threshold_agent(env):
    """
    Geleneksel Kural Tabanlı Algoritma (Rakip)
    Mantık: CPU > %80 ise Artır, CPU < %30 ise Azalt.
    """
    obs, _ = env.reset(seed=SEED)
    # Numpy random seed'i de sabitleyelim ki trafik birebir aynı olsun
    np.random.seed(SEED) 
    
    cpu_history = []
    vm_history = []
    rewards = 0
    
    for _ in range(TEST_STEPS):
        cpu_load = obs[0] # Durumun ilk elemanı CPU
        
        # --- KURAL TABANLI MANTIK ---
        if cpu_load > 80:
            action = 2 # Scale Out (Artır)
        elif cpu_load < 30:
            action = 0 # Scale In (Azalt)
        else:
            action = 1 # Do Nothing (Sabit)
        
        obs, reward, done, _, info = env.step(action)
        
        cpu_history.append(obs[0])
        vm_history.append(obs[1])
        rewards += reward
        
    return cpu_history, vm_history, rewards

def run_rl_agent(env):
    """
    Senin Eğittiğin Yapay Zeka Ajanı
    """
    # Modeli yükle
    try:
        model = DQN.load("dqn_cloud_autoscale")
    except:
        print("Model bulunamadı! Önce train.py çalıştırılmalı.")
        return [], [], 0

    obs, _ = env.reset(seed=SEED)
    np.random.seed(SEED) # AYNI trafik için reset
    
    cpu_history = []
    vm_history = []
    rewards = 0
    
    for _ in range(TEST_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        cpu_history.append(obs[0])
        vm_history.append(obs[1])
        rewards += reward
        
    return cpu_history, vm_history, rewards

# --- KARŞILAŞTIRMA BAŞLIYOR ---
env = CloudEnv()

print("Klasik Yöntem Çalışıyor...")
t_cpu, t_vm, t_reward = run_threshold_agent(env)

print("Yapay Zeka Çalışıyor...")
rl_cpu, rl_vm, rl_reward = run_rl_agent(env)

print("\n--- SONUÇLAR ---")
print(f"Klasik Yöntem Toplam Ödül: {t_reward:.2f}")
print(f"Yapay Zeka Toplam Ödül : {rl_reward:.2f}")

# --- GRAFİK ÇİZİMİ ---
steps = range(TEST_STEPS)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Grafik 1: CPU Kullanımı Karşılaştırması
ax1.plot(steps, t_cpu, 'r--', label='Klasik Yöntem (Threshold)', alpha=0.6)
ax1.plot(steps, rl_cpu, 'b-', label='Yapay Zeka (DQN)', linewidth=2)
ax1.axhline(y=80, color='gray', linestyle=':', label='SLA Sınırı (%80)')
ax1.axhline(y=30, color='gray', linestyle=':', label='Verimlilik Sınırı (%30)')
ax1.set_ylabel('CPU Kullanımı (%)')
ax1.set_title('CPU Performans Karşılaştırması')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Grafik 2: VM Sayısı (Maliyet) Karşılaştırması
ax2.plot(steps, t_vm, 'r--', label='Klasik Yöntem (VM Sayısı)')
ax2.plot(steps, rl_vm, 'b-', label='Yapay Zeka (VM Sayısı)', linewidth=2)
ax2.set_ylabel('Aktif Sunucu Sayısı (VM)')
ax2.set_xlabel('Zaman (Adım)')
ax2.set_title('Kaynak Kullanımı / Maliyet Karşılaştırması')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# --- RAPORLAMA KISMI (Mevcut kodun en altına ekle) ---

def calculate_stats(cpu_data, vm_data):
    # 1. Toplam Maliyet (VM Sayısı * Birim Zaman)
    total_cost = sum(vm_data)
    
    # 2. SLA İhlal Sayısı (CPU > 80 olduğu anlar)
    sla_violations = sum(1 for c in cpu_data if c > 80)
    
    # 3. Ortalama CPU Kullanımı
    avg_cpu = sum(cpu_data) / len(cpu_data)
    
    return total_cost, sla_violations, avg_cpu

t_cost, t_sla, t_cpu_avg = calculate_stats(t_cpu, t_vm)
rl_cost, rl_sla, rl_cpu_avg = calculate_stats(rl_cpu, rl_vm)

print("\n" + "="*40)
print("     DETAYLI PERFORMANS RAPORU")
print("="*40)
print(f"{'Metrik':<20} | {'Klasik (Threshold)':<15} | {'Yapay Zeka (DQN)':<15}")
print("-" * 56)
print(f"{'Toplam Maliyet':<20} | {t_cost:<15.0f} | {rl_cost:<15.0f}")
print(f"{'SLA İhlal Sayısı':<20} | {t_sla:<15} | {rl_sla:<15}")
print(f"{'Ortalama CPU (%)':<20} | {t_cpu_avg:<15.2f} | {rl_cpu_avg:<15.2f}")
print("="*40)