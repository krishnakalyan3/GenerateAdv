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
For generating adversarial examples from a pretrained CNN on cifar10. The pretrained CNN's weights can be loaded from Cifar_pretrained_CudaConvVersion.pkl
<div class="highlight highlight-source-shell"><pre>python genAdv.py -m LBFGS -d cifar10 -n Cifar_pretrained_CudaConvVersion.pkl</pre></div>


<h3> Reference </h3>
<ol type="1">
  <li>Szegedy, Christian, et al. "Intriguing properties of neural networks." arXiv preprint arXiv:1312.6199 (2013).</li>
  <li>Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).</li>
 
</ol>



