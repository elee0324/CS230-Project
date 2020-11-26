## CS230 Final Project: Evaluating the Effectiveness of Adversarial Attacks on Traffic Light Color Classification Deep Neural Networks
Fall 2020

### Abstract
Traffic Light Recognition is a critical task in autonomous and assisted driving, but the literature on adversarial attacks against these models is woefully lacking. We adopt a two-stage traffic light recognition model with two CNNs working to localize and classify traffic lights. We first evaluate adversarial attacks against the classification model. We then attempt to fool the localization model by perturbing a small region of the original input region. Our adversarial attacks were unsuccesful in attacking either model, a major contrast to the one other paper on adversarial attacks against traffic light recognition models. We suggest possible reasons for our contrasting results, and we offer suggestions for additional work to be to assess the robustness of traffic light recognition models.

### Project Video
Below is the link to our virtual presentation:
https://drive.google.com/file/d/1kqGRD9nlS-qRIW1qQ17FuFgBZeEmCBln/view?usp=sharing

### Files
- ```CS230_paper.pdf```: final paper for our project
- ```carla_adversarial.py```: code for localization and classification models and adversarial attack generation and detection (see sources in paper)
