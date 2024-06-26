# ---------- AWS a10040gb GPU INSTANCE COST ----------
# p4d.24xlarge
# - CPU     : Xeon Platinum 8275CL
# - Memory  : 1152GB
# - Storage : 8 x 1TB NVMe SSD
# - GPU     : 8 x 40GB A100
#
# ---------- ON-PREMISE DATACENTER ----------
# CPU     : https://ark.intel.com/content/www/us/en/ark/products/192482/intel-xeon-platinum-8270-processor-35-75m-cache-2-70-ghz.html
# - Intel Xeon Platinum 8270 is compatible with DDR4-2933 memory
# Memory  : https://memory.net/memory-prices/
# - 128GB DDR4-2933 PC4-23466U-L ECC LRDIMM
# Storage : https://www.newegg.com/samsung-1tb-980/p/N82E16820147804
# - Samsung 980 1TB PCIe 3.0 NVMe SSD
# GPU     : https://www.walmart.com/ip/NVIDIA-A100-SXM-Tensor-Core-GPU-40-GB-HBM2/1473647679
# - NVIDIA Tesla A100 40GB SXM
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
    'SSD'           : 819.60,   # USD
    'GPU'           : 79916.00, # USD
    'aws_on_demand' : 32.773,   # USD / HR
    'aws_spot'      : 13.655,   # USD / HR
    'aws_1_yr_res'  : 20.175,   # USD / HR
    'aws_3_yr_res'  : 12.499    # USD / HR
}

# ---------- AWS v10032gb GPU INSTANCE COST ----------
# p3dn.24xlarge
# - CPU     : Xeon Platinum 8175M
# - Memory  : 768GB
# - Storage : 2 x 900GB NVMe SSD
# - GPU     : 8 x 32GB V100
#
# ---------- ON-PREMISE DATACENTER ----------
# CPU     : https://ark.intel.com/content/www/us/en/ark/products/120506/intel-xeon-platinum-8170-processor-35-75m-cache-2-10-ghz.html
# - Intel Xeon Platinum 8170 is compatible with DDR4-2666 memory
# Memory  : https://memory.net/memory-prices/
# - 128GB DDR4-2666 PC4-21300V-L ECC LRDIMM
# Storage : https://www.newegg.com/samsung-1tb-980/p/N82E16820147804
# - Samsung 980 1TB PCIe 3.0 NVMe SSD
# GPU     : https://www.amazon.com/NVIDIA-Tesla-Volta-Accelerator-Graphics/dp/B07JVNHFFX
# - NVIDIA Tesla V100 32GB PCIe
#
# ---------- CLOUD RENTAL ----------
# AWS EC2 USD/HR : https://instances.vantage.sh/aws/ec2/p3dn.24xlarge
# - On Demand
# - Spot
# - 1 Yr Reserved
# - 3 Yr Reserved
aws_p3dn_24xlarge_cost = {
    'CPU'           : 7405.00,  # USD
    'DRAM'          : 2928.00,  # USD
    'SSD'           : 204.90,   # USD
    'GPU'           : 27519.52, # USD
    'aws_on_demand' : 31.212,   # USD / HR
    'aws_spot'      : 12.468,   # USD / HR
    'aws_1_yr_res'  : 19.215,   # USD / HR
    'aws_3_yr_res'  : 10.416    # USD / HR
}
