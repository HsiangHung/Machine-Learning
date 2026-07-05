
# Transformers For Recommendation Sys

Transformers4Rec is a flexible and efficient library for sequential and **session-based recommendation** and can work with PyTorch.

Input data is typically a sequence of interactions such as items that are browsed in a web session or items put in a cart. The library helps you process and model the interactions so that you can output better recommendations for the next item.


## Sequential and Session-based recommendation with Transformers4Rec

Traditional recommendation algorithms usually ignore the temporal dynamics and the sequence of interactions when trying to model user behavior. Generally, the next user interaction is related to the sequence of the user's previous choices. In some cases, it might be a repeated purchase or song play. User interests can also suffer from interest drift because preferences can change over time. Those challenges are addressed by the sequential recommendation task.