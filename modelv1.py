import joblib
import pandas as pd


# ===========================
# Load Trained Model
# ===========================

MODEL_PATH = "benchmark_model.pkl"

model = joblib.load(MODEL_PATH)


# ===========================
# Prediction Function
# ===========================

def benchmark_system(
    cpu_model,
    cpu_generation,
    cpu_cores,
    cpu_threads,
    cpu_base,
    cpu_boost,
    cpu_cache,
    cpu_tdp,
    igpu,
    igpu_score,
    dgpu,
    has_gpu,
    dgpu_score,
    vram,
    gpu_cores,
    gpu_compute,
    gpu_bus,
    ram,
    ram_ddr,
    ram_speed,
    storage,
    storage_speed
):

    data = pd.DataFrame([{

        "CPU_Model": cpu_model,
        "CPU_Generation": cpu_generation,
        "CPU_Cores": cpu_cores,
        "CPU_Threads": cpu_threads,
        "CPU_Base_GHz": cpu_base,
        "CPU_Boost_GHz": cpu_boost,
        "CPU_Cache_MB": cpu_cache,
        "CPU_TDP_W": cpu_tdp,

        "Integrated_GPU": igpu,
        "Integrated_GPU_Score": igpu_score,

        "Dedicated_GPU": dgpu,
        "Has_Dedicated_GPU": has_gpu,
        "Dedicated_GPU_Score": dgpu_score,
        "GPU_VRAM_GB": vram,
        "GPU_Cores": gpu_cores,
        "GPU_Compute_Type": gpu_compute,
        "GPU_Memory_Bus_Bit": gpu_bus,

        "RAM_GB": ram,
        "RAM_DDR": ram_ddr,
        "RAM_Speed_MHz": ram_speed,

        "Storage_Type": storage,
        "Storage_Read_MBps": storage_speed

    }])

    score = float(model.predict(data)[0])

    if score < 30:
        tier = "Low End"

    elif score < 55:
        tier = "Entry Level"

    elif score < 75:
        tier = "Mid Range"

    elif score < 90:
        tier = "High End"

    else:
        tier = "Enthusiast"

    recommendations = []

    if ram < 8:
        recommendations.append("Upgrade RAM to at least 8 GB.")

    if ram_speed < 2400:
        recommendations.append("Higher RAM speed will improve performance.")

    if storage == "HDD":
        recommendations.append("Upgrade to SSD or NVMe.")

    if dgpu == "NoGPU":
        recommendations.append("Install a dedicated graphics card.")

    if vram < 4 and dgpu != "NoGPU":
        recommendations.append("A GPU with 4GB+ VRAM is recommended.")

    if cpu_cores < 4:
        recommendations.append("Consider upgrading to a 4+ core processor.")

    return {
        "Benchmark Score": round(score,2),
        "Performance Tier": tier,
        "Recommendations": recommendations
    }


# =====================================
# Example
# =====================================

result = benchmark_system(

    cpu_model="i5-2400",
    cpu_generation=2,
    cpu_cores=4,
    cpu_threads=4,
    cpu_base=3.1,
    cpu_boost=3.4,
    cpu_cache=6,
    cpu_tdp=95,

    igpu="Intel HD 2000",
    igpu_score=320,

    dgpu="R5 240",
    has_gpu=1,
    dgpu_score=1800,
    vram=1,
    gpu_cores=320,
    gpu_compute="Stream",
    gpu_bus=128,

    ram=32,
    ram_ddr="DDR3",
    ram_speed=1600,

    storage="SATA SSD",
    storage_speed=550

)

print("\n========== AI System Benchmark ==========\n")

print("Benchmark Score :", result["Benchmark Score"])
print("Performance Tier:", result["Performance Tier"])

print("\nRecommendations")

if len(result["Recommendations"]) == 0:
    print("No upgrades required.")

else:
    for recommendation in result["Recommendations"]:
        print("•", recommendation)