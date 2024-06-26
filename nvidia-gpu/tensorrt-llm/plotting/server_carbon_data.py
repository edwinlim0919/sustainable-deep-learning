# Building an AWS EC2 Carbon Emissions Dataset (Benjamin Davy): https://medium.com/teads-engineering/building-an-aws-ec2-carbon-emissions-dataset-3f0fd76c98ac
# Dataset spreadsheet: https://docs.google.com/spreadsheets/d/1DqYgQnEDLQVQm5acMAhLgHLD8xXCG9BIrk-_Nv6jF3k/

# ---------- STANDARD SERVER CARBON BREAKDOWN ----------
# Embodied carbon data for Dell PowerEdge R740 (kgCO2eq)
dell_poweredge_r740_carbon = {
    'SSD'        : {
        0 : 3379,       # 8 * 3.84TB Solid State Drives
        1 : 64          # 1 * 400GB Solid State Drive
    },
    'DRAM'       : 533, # 12 * 32GB DIMMs Memory
    'PWB'        : 109, # Mainboard PWB
    'Networking' : 59,  # Riser card 1 - Riser card 2 - Riser card 3 - Ethernet card - HDD Controller - Q-logic - Intel Ethernet X710
    'CPU'        : 47,  # 2 * Xeon CPUs with housing
    'Chassis'    : 34,
    'PSU'        : 30,
    'IO'         : 20,  # Mainboard Connectors - Transport
    'Fans'       : 13
}
# Operational carbon data for Dell PowerEdge R740 (TODO)

# ---------- AWS GPU INSTANCE CARBON OVERALL ----------
# p4d.24xlarge
# - CPU     : Xeon Platinum 8275CL
# - Memory  : 1152GB
# - Storage : 8 x 1TB NVMe SSD
# - GPU     : 8 x 40GB A100
aws_p4d_24xlarge_carbon = {
    'pkg_power_0'       : 57.93,   # Watts
    'pkg_power_10'      : 175.53,  # Watts
    'pkg_power_50'      : 448.31,  # Watts
    'pkg_power_100'     : 626.76,  # Watts
    'ram_power_0'       : 219.30,  # Watts
    'ram_power_10'      : 397.56,  # Watts
    'ram_power_50'      : 830.22,  # Watts
    'ram_power_100'     : 1262.88, # Watts
    'gpu_power_0'       : 371.6,   # Watts
    'gpu_power_10'      : 1018.4,  # Watts
    'gpu_power_50'      : 2407.2,  # Watts
    'gpu_power_100'     : 3259.3,  # Watts
    'memory_embodied'   : 1575.5,  # kgCO2eq
    'platform_embodied' : 800.0,   # kgCO2eq
    'gpu_embodied'      : 1200.0,  # kgCO2eq
    'cpu_embodied'      : 100.0    # kgCO2eq
}

# p3dn.24xlarge
# - CPU     : Xeon Platinum 8175M
# - Memory  : 768GB
# - Storage : 2 x 900GB NVMe SSD
# - GPU     : 8 x 32GB V100
aws_p3dn_24xlarge_carbon = {
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
