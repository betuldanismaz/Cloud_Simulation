from stable_baselines3 import DQN
from cloud_env import CloudEnv # Kendi yazdığımız ortamı içeri aktarıyoruz
from stable_baselines3 import DQN
from cloud_env import CloudEnv

# 1. Ortamı Başlat
env = CloudEnv()

# 2. Modeli Tanımla (GÜNCELLENDİ)
# learning_rate: 0.0001'den 0.001'e çektik (Daha hızlı öğrensin)
# gamma: Gelecekteki ödüllere ne kadar önem verdiği (0.99 standarttır)
model = DQN("MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.001, 
            gamma=0.99,
            buffer_size=100000, # Hafızasını genişlettik
            exploration_fraction=0.2 # Keşfetme oranını ayarladık
            )

print("--- EĞİTİM BAŞLIYOR (V2 - Gelişmiş) ---")
print("Bu işlem biraz daha uzun sürecek (Yaklaşık 1-2 dakika)...")

# 3. Modeli Eğit (GÜNCELLENDİ)
# 10.000 adım yerine 100.000 adım! (10 Kat daha fazla deneyim)
model.learn(total_timesteps=100000)

print("--- EĞİTİM TAMAMLANDI ---")

# 4. Modeli Kaydet
model.save("dqn_cloud_autoscale")
print("Yeni Model Kaydedildi.")

print("Model 'dqn_cloud_autoscale.zip' olarak kaydedildi.")

# --- TEST AŞAMASI ---
# Bakalım eğitilen model gerçekten akıllıca davranıyor mu?

print("\n--- TEST SONUÇLARI ---")
obs, _ = env.reset()
total_reward = 0

for i in range(20): # 20 adımlık bir test yapalım
    # predict: Modele "Bu durumda ne yapayım?" diye soruyoruz.
    # deterministic=True: En garantici (en iyi bildiği) yolu seç demek.
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    action_name = ["Azalt", "Sabit", "ArtTır"][action]
    
    print(f"Adım {i+1}: Durum [CPU %{obs[0]:.1f} | VM {int(obs[1])}] -> Karar: {action_name} -> Ödül: {reward}")

print(f"\nToplam Test Ödülü: {total_reward}")