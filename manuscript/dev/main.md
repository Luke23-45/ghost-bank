We evaluated the incremental class learning on the CIFAR-100 dataset with a bounded active exemplar budget. Training runs through tasks $t=0,\ldots,T-1$; each task
brings a disjoint set of $C_{\mathrm{new}}$ classes, and evaluation after
task $t$ covers all classes seen so far, without task identity.  In the
main experiments, $T=10$ and $C_{\mathrm{new}}=10$.  We permute the CIFAR-100
classes with split seed 13 and remap them to contiguous internal labels.  From
each training class, we hold out 30 images for probing and 20 for validation;
none enter replay storage.