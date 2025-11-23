import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class CloudEnv(gym.Env):
    def __init__(self):
        super(CloudEnv, self).__init__()
        # --- AYARLAR ---
        self.min_vm = 1
        self.max_vm = 20
        self.vm_capacity = 100
        
        # --- EYLEM UZAYI ---
        # 0: Azalt, 1: Sabit Kal, 2: Artır
        self.action_space = spaces.Discrete(3)
        
        # --- GÖZLEM UZAYI ---
        # [CPU (0-100), VM Sayısı (1-20), Gelen İstek (0-5000)]
        self.observation_space = spaces.Box(
            low=np.array([0, self.min_vm, 0]), 
            high=np.array([100, self.max_vm, 5000]),
            dtype=np.float32
        )
        
        self.max_steps = 100 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_vm = 1
        self.current_step = 0
        
        # Başlangıç Trafiği
        self.current_requests = 100 
        self.current_cpu = (self.current_requests / (self.current_vm * self.vm_capacity)) * 100
        
        self.state = np.array([self.current_cpu, self.current_vm, self.current_requests], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # 1. AKSİYON UYGULA
        if action == 2: # Scale Out
            self.current_vm = min(self.current_vm + 1, self.max_vm)
        elif action == 0: # Scale In
            self.current_vm = max(self.current_vm - 1, self.min_vm)
        
        # 2. TRAFİK SİMÜLASYONU (SİNÜS DALGASI)
        base_load = 500
        amplitude = 400 
        noise = np.random.normal(0, 20) 
        
        time_factor = (self.current_step / self.max_steps) * 2 * math.pi
        self.current_requests = base_load + (amplitude * math.sin(time_factor)) + noise
        self.current_requests = max(10, self.current_requests)

        # 3. CPU HESABI
        total_capacity = self.current_vm * self.vm_capacity
        self.current_cpu = (self.current_requests / total_capacity) * 100
        self.current_cpu = min(100, self.current_cpu)

        # 4. ÖDÜL SİSTEMİ (YENİ VE DÜZELTİLMİŞ)
        reward = 0
        
        # Kural 1: SLA İhlali
        if self.current_cpu > 80: 
            reward -= 100
            
        # Kural 2: Kaynak Maliyeti
        reward -= (self.current_vm * 1) 
        
        # Kural 3: Verimsizlik Cezası (BOŞA ÇALIŞMA!)
        # CPU %40'ın altındaysa ceza artar
        if self.current_cpu < 40:
            wasted_resource = 40 - self.current_cpu
            reward -= (wasted_resource * 2) 
            
        # Kural 4: Ölçekleme Bonusu (Scale In Teşviki)
        if action == 0 and self.current_cpu < 75:
            reward += 10

        # 5. BİTİŞ KONTROLÜ VE DURUM GÜNCELLEME
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        self.state = np.array([self.current_cpu, self.current_vm, self.current_requests], dtype=np.float32)
        info = {"request_count": self.current_requests, "cpu_load": self.current_cpu}
        
        # İŞTE EKSİK OLAN KRİTİK SATIR BURASIYDI:
        return self.state, reward, terminated, False, info