
import joblib
import pandas as pd


# =========================
# CPU DATABASE
# =========================
class CPUDatabase:

    def __init__(self, file):
        self.cpu_db = pd.read_csv(file)

    def get_cpu(self, cpu_model):
        cpu = self.cpu_db[
            self.cpu_db["CPU_Model"].str.lower() == cpu_model.lower()
        ]

        if cpu.empty:
            raise ValueError(f"CPU '{cpu_model}' not found.")

        return cpu.iloc[0]

    def show_cpus(self):
        print("\nAvailable CPUs: match the model or similar as shown below.")
        cpus = self.cpu_db["CPU_Model"].tolist()
        print(cpus)


# =========================
# GPU DATABASE
# =========================
class GPUDatabase:

    def __init__(self, file):
        self.gpu_db = pd.read_csv(file)

    def get_gpu(self, gpu_model):
        gpu = self.gpu_db[
            self.gpu_db["Dedicated_GPU"].str.lower() == gpu_model.lower()
        ]

        if gpu.empty:
            raise ValueError(f"GPU '{gpu_model}' not found.")

        return gpu.iloc[0]

    def show_gpus(self):
        print("\nAvailable GPUs: match the model or similar as shown below.")
        gpus = self.gpu_db["Dedicated_GPU"].tolist()
        print(gpus)


# =========================
# BENCHMARK SYSTEM
# =========================
class BenchmarkSystem:

    def __init__(self, model_file, cpu_file, gpu_file):

        # Load ML model
        self.model = joblib.load(model_file)

        # Create database objects
        self.cpu_database = CPUDatabase(cpu_file)
        self.gpu_database = GPUDatabase(gpu_file)

    def calculate_tier(self, score):

        if score < 30:
            return "Low End"

        elif score < 55:
            return "Entry Level"

        elif score < 75:
            return "Mid Range"

        elif score < 90:
            return "High End"

        else:
            return "Enthusiast"

    def benchmark_system(
            self,
            cpu_model,
            gpu_model,
            ram,
            ram_ddr,
            ram_speed,
            storage,
            storage_speed):

        # Get CPU information
        cpu = self.cpu_database.get_cpu(cpu_model)

        # Get GPU information
        gpu = self.gpu_database.get_gpu(gpu_model)

        # Create input data for ML model
        data = pd.DataFrame([{

            "CPU_Model": cpu["CPU_Model"],
            "CPU_Generation": cpu["CPU_Generation"],
            "CPU_Cores": cpu["CPU_Cores"],
            "CPU_Threads": cpu["CPU_Threads"],
            "CPU_Base_GHz": cpu["CPU_Base_GHz"],
            "CPU_Boost_GHz": cpu["CPU_Boost_GHz"],
            "CPU_Cache_MB": cpu["CPU_Cache_MB"],
            "CPU_TDP_W": cpu["CPU_TDP_W"],

            "Integrated_GPU": cpu["Integrated_GPU"],
            "Integrated_GPU_Score": cpu["Integrated_GPU_Score"],

            "Dedicated_GPU": gpu["Dedicated_GPU"],
            "Has_Dedicated_GPU":
                0 if gpu["Dedicated_GPU"] == "NoGPU" else 1,

            "Dedicated_GPU_Score": gpu["Dedicated_GPU_Score"],
            "GPU_VRAM_GB": gpu["GPU_VRAM_GB"],
            "GPU_Cores": gpu["GPU_Cores"],
            "GPU_Compute_Type": gpu["GPU_Compute_Type"],
            "GPU_Memory_Bus_Bit": gpu["GPU_Memory_Bus_Bit"],

            "RAM_GB": ram,
            "RAM_DDR": ram_ddr,
            "RAM_Speed_MHz": ram_speed,

            "Storage_Type": storage,
            "Storage_Read_MBps": storage_speed

        }])

        # Predict benchmark score
        score = float(self.model.predict(data)[0])

        # Determine tier
        tier = self.calculate_tier(score)

        return score, tier


# =========================
# MAIN PROGRAM
# =========================

benchmark = BenchmarkSystem(
    model_file="benchmark_model.pkl",
    cpu_file="cpu_specs.csv",
    gpu_file="gpu_specs.csv"
)


print("=" * 55)
print("        PC PERFORMANCE BENCHMARK SYSTEM")
print("=" * 55)

print("\nThis program will analyze your PC.")
print("Please enter your hardware information carefully.")
print("The program will use your information to calculate")
print("an estimated benchmark score.\n")


# =========================
# CPU INPUT
# =========================

benchmark.cpu_database.show_cpus()

while True:

    cpu_model = input(
        "\nEnter your CPU model exactly as shown above: "
    ).strip()

    try:
        benchmark.cpu_database.get_cpu(cpu_model)
        break

    except ValueError:
        print("\nERROR: CPU not found.")
        print("Please select a CPU from the available list.")


# =========================
# GPU INPUT
# =========================

benchmark.gpu_database.show_gpus()

while True:

    gpu_model = input(
        "\nEnter your dedicated GPU model exactly as shown above: "
    ).strip()

    try:
        benchmark.gpu_database.get_gpu(gpu_model)
        break

    except ValueError:
        print("\nERROR: GPU not found.")
        print("Please select a GPU from the available list.")


# =========================
# RAM INPUT
# =========================

print("\n" + "-" * 55)
print("RAM INFORMATION")
print("-" * 55)

print("Example: 8, 16, 32")

while True:

    try:
        ram = int(input("Enter your RAM size in GB: "))

        if ram <= 0:
            print("RAM must be greater than 0.")
            continue

        break

    except ValueError:
        print("Please enter a valid number.")


ram_ddr = input(
    "Enter your RAM type (Example: DDR3, DDR4, DDR5): "
).strip()


# =========================
# RAM SPEED
# =========================

print("\nRAM Speed Examples:")
print("DDR3 → 1333 / 1600 MHz")
print("DDR4 → 2400 / 2666 / 3200 MHz")
print("DDR5 → 4800 / 5200 / 5600 MHz")

while True:

    try:
        ram_speed = int(
            input("Enter your RAM speed in MHz: ")
        )

        if ram_speed <= 0:
            print("RAM speed must be greater than 0.")
            continue

        break

    except ValueError:
        print("Please enter a valid number.")


# =========================
# STORAGE INPUT
# =========================

print("\n" + "-" * 55)
print("STORAGE INFORMATION")
print("-" * 55)

print("Examples:")
print("SATA SSD")
print("NVMe SSD")
print("HDD")

storage = input(
    "Enter your storage type: "
).strip()


# =========================
# STORAGE SPEED
# =========================

print("\nStorage Read Speed Examples:")
print("HDD      → around 80-200 MB/s")
print("SATA SSD → around 450-550 MB/s")
print("NVMe SSD → around 1500-7000+ MB/s")

while True:

    try:
        storage_speed = int(
            input("Enter your storage read speed in MB/s: ")
        )

        if storage_speed <= 0:
            print("Storage speed must be greater than 0.")
            continue

        break

    except ValueError:
        print("Please enter a valid number.")


# =========================
# SHOW USER INPUT
# =========================

print("\n" + "=" * 55)
print("YOUR PC CONFIGURATION")
print("=" * 55)

print("CPU          :", cpu_model)
print("GPU          :", gpu_model)
print("RAM          :", ram, "GB")
print("RAM Type     :", ram_ddr)
print("RAM Speed    :", ram_speed, "MHz")
print("Storage      :", storage)
print("Storage Speed:", storage_speed, "MB/s")


# =========================
# RUN BENCHMARK
# =========================

print("\nCalculating benchmark...")
print("Please wait...\n")

try:

    score, tier = benchmark.benchmark_system(

        cpu_model=cpu_model,
        gpu_model=gpu_model,

        ram=ram,
        ram_ddr=ram_ddr,
        ram_speed=ram_speed,

        storage=storage,
        storage_speed=storage_speed
    )

    # =========================
    # RESULT
    # =========================

    print("=" * 55)
    print("              BENCHMARK RESULT")
    print("=" * 55)

    print(f"\nBenchmark Score : {score:.2f}")
    print(f"Performance Tier: {tier}")

    print("\n" + "=" * 55)

except Exception as e:

    print("\nAn error occurred:")
    print(e)

