# GenerateAdv
Python codes for two popular methods for generating adversarial examples: LBFGS [<a href="https://arxiv.org/abs/1312.6199">1</a>]
and fast gradient sign [<a href ="https://arxiv.org/abs/1412.6572">2</a>].

<h3>Requirements </h3>

<ul style="list-style-type:disc">
  <li>Python 2.7</li>
  <li> <a href = "http://deeplearning.net/software/theano/">Theano</a></li>
  <li><a href = "https://lasagne.readthedocs.io/en/latest/"> Lasagne</a> </li>

</ul>

<h3>Instruction</h3>
For generating adversarial examples from a pretrained CNN on cifar10 and mnist, one can load one of the two provided pretrained CNNs: one for cifar10 and the other for mnist are Cifar_pretrained_CudaConvVersion.pkl and MNIST_pretrained_CudaConvVersion.pkl, respectively. The architectures of these CNNs are identical [<a href ='https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-18pct.cfg'>3</a>]. In the following, the command for generating adversarial examples is presented, where -m indicates method for generating adversarial examples (two options: LBFGS, FastSign), -d takes the name of the desired dataset (two options: cifar10, mnist) , and -n indicates the weights of a pretrained CNN.
<div class="highlight highlight-source-shell"><pre>python genAdv.py -m LBFGS -d cifar10 -n Cifar_pretrained_CudaConvVersion.pkl</pre></div>


<h3> Reference </h3>
<ol type="1">
  <li>Szegedy, Christian, et al. "Intriguing properties of neural networks." arXiv preprint arXiv:1312.6199 (2013).</li>
  <li>Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).</li>

 
</ol>



