
# Towards Interpretable Multi-Task Learning Using Bilevel Programming
Francesco Alesiani, Shujian Yu, Ammar Shaker, and Wenzhe Yin

The code repository for the ECML/PKDD 2020 paper [Towards Interpretable Multi-Task Learning Using Bilevel Programming](https://arxiv.org/abs/2009.05483).

Interpretable Multi-Task Learning can be expressed as learning a sparse graph of the task relationship based on the prediction performance of the learned models. Since many natural phenomenon exhibit sparse structures, enforcing sparsity on learned models reveals the underlying task relationship. Moreover, different sparsification degrees from a fully connected graph uncover various types of structures, like cliques, trees, lines, clusters or fully disconnected graphs. In this paper, we propose a bilevel formulation of multi-task learning that induces sparse graphs, thus, revealing the underlying task relationships, and an efficient method for its computation. We show empirically how the induced sparse graph improves the interpretability of the learned models and their re- lationship on synthetic and real data, without sacrificing generalization performance.


## Citations

```
@inproceedings{DBLP:conf/pkdd/AlesianiYSY20,
  author    = {Francesco Alesiani and
               Shujian Yu and
               Ammar Shaker and
               Wenzhe Yin},
  editor    = {Frank Hutter and
               Kristian Kersting and
               Jefrey Lijffijt and
               Isabel Valera},
  title     = {Towards Interpretable Multi-task Learning Using Bilevel Programming},
  booktitle = {Machine Learning and Knowledge Discovery in Databases - European Conference,
               {ECML} {PKDD} 2020, Ghent, Belgium, September 14-18, 2020, Proceedings,
               Part {II}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12458},
  pages     = {593--608},
  publisher = {Springer},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-67661-2\_35},
  doi       = {10.1007/978-3-030-67661-2\_35},
  timestamp = {Sun, 02 Oct 2022 16:13:41 +0200},
  biburl    = {https://dblp.org/rec/conf/pkdd/AlesianiYSY20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


