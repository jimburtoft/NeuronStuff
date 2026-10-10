"""Write the --network-interfaces JSON for an all-EFA launch.

  python make_efa_nics.py <num_cards> <subnet-id> <sg-id> > net.json
  trn2.48xlarge: 16 cards; p6-b200.48xlarge: 8; p5.48xlarge: 32.
Card 0 is "efa" (carries IP traffic, attach an EIP to it); the rest are "efa-only".
"""
import json, sys
n, subnet, sg = int(sys.argv[1]), sys.argv[2], sys.argv[3]
print(json.dumps([{"NetworkCardIndex": i, "DeviceIndex": 0 if i == 0 else 1,
                   "InterfaceType": "efa" if i == 0 else "efa-only", "SubnetId": subnet,
                   "Groups": [sg], "DeleteOnTermination": True} for i in range(n)], indent=1))
