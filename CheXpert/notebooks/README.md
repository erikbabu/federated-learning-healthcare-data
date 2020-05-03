# Launch jupyter notebook from lab machine

1. ssh into lab machine (e.g. ray04)

2. run ```jupyter-lab --no-browser --port=8888```

3. copy the link generated.

4. in new **local** terminal window, run ```ssh -N -f -L localhost:8888:localhost:8888 eb1816@ray04.doc.ic.ac.uk```

5. paste generated link in local browser
