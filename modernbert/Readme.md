# ModernBERT

These examples show how to run a quick benchmark on a sequence classification model.

In one case we use Optimum Neuron.  In another, we are using the Neuronx Transformers library to just trace the model.  We can do this because the model is small enough to fit in a single core.

Note that sequence length will affect the throughput (smaller lengths can process faster)

Don't use the workers code as an example of multi-threading.  Each of these scripts only run on a single Neuron core, so in your code you want to have multiple instances running.

The autocast setting ("none" by default, "matmult" for faster) affects how the Neuron devices do the math internally,  "matmult" is about 2x as fast, but could affect your results after a few decimal places.

Batching on transformers doesn't seem to improve your speed any.

You should be able to trace and run multiple sequence lengths and pick the correct one at runtime, but I don't have an example for that.

This is tested on Inferentia2.  You should be able to run it on any Neuroncore v2 or later device (Trainium1, Trainium2, etc.)

Inferentia1 didn't work because it supports Pytorch 1.13.1.  I could only load an older version of transformers ( 4.38.2) that didn't recognize modernbert.  If you can load the graph somehow and trace it you may be able to run it.

Optimum Neuron runs 2x faster for the same sequence length , batch size, and auto-cast setting.  The Optimum Neuron library supports models with the Neuronx Distributed library.  I'm not sure if that is the difference or if they are compiling with a different setting.
