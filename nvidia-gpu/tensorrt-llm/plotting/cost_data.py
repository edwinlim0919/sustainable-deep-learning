# ---------- AWS a10040gb GPU INSTANCE COST OVERALL ----------
# p4d.24xlarge
# - CPU     : Xeon Platinum 8275CL
# - Memory  : 1152GB
# - Storage : 8TB NVMe SSD
# - GPU     : 8 x 40GB A100
#
# ---------- ON-PREMISE DATACENTER ----------
# CPU USD     : https://ark.intel.com/content/www/us/en/ark/products/192482/intel-xeon-platinum-8270-processor-35-75m-cache-2-70-ghz.html
# - Intel Xeon Platinum 8270 is compatible with DDR4-2933 memory
# Memory USD  : https://memory.net/memory-prices/
# - 128GB DDR4-2933 PC4-23466U-L ECC LRDIMM
# Storage USD : https://www.newegg.com/intel-dc-p4510-8tb/p/1Z4-009F-00083
# - Intel DC P4510 8TB PCIe NVMe SSD
# GPU USD     : https://www.amazon.com/NVIDIA-Ampere-Graphics-Processor-Accelerator/dp/B08X13X6HF
# - NVIDIA Tesla A100 40GB PCIe
#
# ---------- CLOUD RENTAL ----------
# AWS EC2 USD/HR : https://instances.vantage.sh/aws/ec2/p4d.24xlarge
# - On Demand
# - Spot
# - 1 Yr Reserved
# - 3 Yr Reserved
aws_p4d_24xlarge_cost = {
    'CPU'           : 8477.00,  # USD
    'DRAM'          : 4932.00,  # USD
    'SSD'           : 1350.55,  # USD
    'GPU'           : 63679.92, # USD
    'aws_on_demand' : 32.773,   # USD / HR
    'aws_spot'      : 13.655,   # USD / HR
    'aws_1_yr_res'  : 20.175,   # USD / HR
    'aws_3_yr_res'  : 12.499    # USD / HR
}

# p3dn.24xlarge
# - CPU     : Xeon Platinum 8175M
# - Memory  : 768GB
# - Storage : 2 x 900GB NVMe SSD
# - GPU     : 8 x 32GB V100
aws_p3dn_24xlarge_cost = {
    'pkg_power_0'       : 57.88,  # Watts
    'pkg_power_10'      : 146.63, # Watts
    'pkg_power_50'      : 343.70, # Watts
    'pkg_power_100'     : 477.98, # Watts
    'ram_power_0'       : 115.62, # Watts
    'ram_power_10'      : 184.78, # Watts
    'ram_power_50'      : 476.20, # Watts
    'ram_power_100'     : 767.62, # Watts
    'gpu_power_0'       : 278.7,  # Watts
    'gpu_power_10'      : 763.8,  # Watts
    'gpu_power_50'      : 1805.4, # Watts
    'gpu_power_100'     : 2444.5, # Watts
    'memory_embodied'   : 1042.9, # kgCO2eq
    'platform_embodied' : 200.0,  # kgCO2eq
    'gpu_embodied'      : 1200.0, # kgCO2eq
    'cpu_embodied'      : 100.0   # kgCO2eq
}
