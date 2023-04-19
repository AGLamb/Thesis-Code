Space. Time. Continuity. Three words that tend to be grouped together in the realm of physics. They are the pillars 
of reality and the base upon which we model a large proportion of physical processes. While these processes are 
deterministic (ignoring quantum mechanics), they can be nearly impossible to analyse due to the complexity of 
computing of all relevant information. Thus, approximating them is a far easier solution that bring forth 
non-trivial implications. If model correctly, we can describe our process and provide forecast with a certain level 
of confidence. Created by Boltzmann, Maxwell and Gibbs, the entire field of statistical mechanics is devoted to the 
use of probability theory and statistics to explain the macroscopic behaviour of nature. Modelling these complex 
processes can a titanic endeavour. Abstracting away from the real world is necessary to create mathematical models 
that can be scaled and easy to interpret. To do this, we make use of different tools such as dynamical systems and 
statistical models. While entire field of their own are devoted to these mathematical constructs, both can be used 
in unison to understand physical processes. We focus on the latter as a first approximation to the earlier and build 
an intuition of their connection. 

Dynamical systems are mathematical models that describes how a system changes over time. These systems are 
ubiquitous in science and approximations to them are crucial for several reasons. Firstly, dynamical systems in 
various scientific and engineering fields are often highly complex and challenging to model accurately due to the 
inherent nonlinearities, uncertainties, and multi-scale interactions involved. Developing exact analytical solutions 
or simulations for such systems can be computationally expensive, time-consuming, and sometimes even infeasible. 
Therefore, approximations are necessary to make these systems more tractable and practical to study and analyse. 
Secondly, approximations can provide insights and understanding of the underlying dynamics of complex systems. By 
simplifying the complex dynamics of a system into a more manageable form, approximations can reveal the key 
mechanisms and behaviours that drive the system's behaviour. This can help researchers and practitioners gain deeper 
insights into the system's behaviour, predict its future states, and make informed decisions for practical 
applications. Thirdly, approximations can facilitate the development of practical tools and techniques for 
real-world applications. In many cases, exact solutions or simulations may not be feasible for real-time 
decision-making or practical implementation due to the constraints of time, computational resources, and data 
availability. Approximations, on the other hand, can provide computationally efficient and scalable methods that can 
be practically applied in real-world scenarios. For example, in the context of Lagrangian particle movement in the 
atmosphere, the statistical model proposed in the essay can offer a more efficient and scalable approach compared to 
physical simulations, allowing for faster and cost-effective prediction of particle movement, and informing policy 
decisions for air pollution management. Furthermore, approximations can enable the development of predictive models 
that can be used for scenario analysis, risk assessment, and policy evaluation. By providing simplified yet accurate 
representations of complex dynamical systems, approximations can allow researchers and decision-makers to explore 
different scenarios, assess risks, and evaluate the effectiveness of different policy interventions. This can have 
significant practical implications in fields such as environmental management, public health, and disaster response, 
where timely and informed decision-making is critical. Finally, approximations can foster interdisciplinary research 
and collaboration across different fields of study. Many complex systems, such as those involving the atmosphere, 
involve multi-disciplinary approaches that require expertise from diverse fields, including mathematics, physics, 
statistics, and domain-specific sciences. Approximations can provide a common language and framework for researchers 
from different disciplines to collaborate and develop a holistic understanding of the system's dynamics. In the 
context of the proposed statistical model for Lagrangian particle movement, the model can bridge the gap between 
atmospheric science, statistical modelling, and policymaking, promoting interdisciplinary research and collaboration 
in addressing complex environmental challenges.

Instead of using this approach, we turn to spatial data analysis and statistical models to simplify the nature of 
the systems and their computational requirements. Spatiotemporal modelling has been a field of vast interest in the 
last few decades. However, spatiotemporal data, such as data related to climate, ecology, epidemiology, 
transportation, and social networks, often possess unique characteristics that can pose difficulties for statistical 
analysis. It is simple to see that incorporating the notion of physical distance on a statistical model can be 
complicated. This specific type of data tends to be heterogeneous and have multidimensional dependence. The first 
meaning that the data points may exhibit significant variability in space and time. For example, climate data may 
have variations in temperature, precipitation, and other variables across different locations and time periods. This
heterogeneity can make it challenging to model the data using standard statistical methods that assume homogeneity 
and may require specialized techniques to handle the spatial and temporal variability appropriately. On the other 
hand, the multidimensional dependence can be divided in spatial dependence and temporal dependence. The earlier 
means that data points that are close in space tend to have similar values. This spatial dependence violates the 
independence assumption of many statistical methods, which assumes that data points are independent and identically 
distributed (i.i.d.). Ignoring spatial dependence can lead to biased estimates and inaccurate inferences. 
Incorporating spatial dependence into statistical models requires specialized techniques such as spatial 
autocorrelation models or spatial filtering methods. The latter means that data points at different time points may 
be correlated or exhibit patterns over time. This temporal dependence violates the independence assumption of many 
statistical methods for time series analysis, which assumes that data points are independent. Ignoring temporal 
dependence can lead to biased estimates and inaccurate forecasts. Incorporating temporal dependence into 
statistical models may require time series analysis techniques such as autoregressive integrated moving average 
(ARIMA) models or state-space models. Adding to the already complicated nature of the data, the dataset tends to be 
of significant size, which increases the complexity of the analysis, modelling and interpretation. The advent of 
computation has help us cope with the increasing size of these data sets, however traditional statistical methods 
may not be scalable or efficient enough to hand big spatiotemporal data and specialized techniques such as 
distributed computing, parallel processing, or machine learning algorithms may be required. 

This thesis proposes a statistical model to analyse Lagrangian particle movement in the atmosphere, which can be 
used to predict the movement of particles over time and space, and to identify key factors that influence particle 
movement. The model is designed to be more efficient and accurate than traditional physical simulations and can be 
applied to a wide range of scenarios, from local air pollution to global climate change. The approach mixes the 
characteristics of Eulerian and Lagrangian atmospheric models in using Dynamic Smooth Transition Spatial 
Autoregressive Model (D-STSAR). The movement of particles in the atmosphere plays a crucial role in determining air 
quality, climate change, and weather patterns. Particles can come from a variety of sources, such as natural 
processes, human activity, or industrial processes, and their movement can be influenced by a wide range of factors,
including wind patterns, temperature, and humidity. Understanding the movement of particles in the atmosphere is 
therefore essential for predicting and mitigating the effects of air pollution, as well as for understanding the 
broader impacts of climate change. Traditional methods for tracking particle movement in the atmosphere have relied 
on physical simulations, which can be time-consuming and expensive, and may not always accurately reflect the 
complex interactions between particles and their environment. In recent years, there has been growing interest in 
using statistical models to better understand the movement of particles in the atmosphere. In the second chapter, 
we go through the relevant literature on spatiotemporal modelling and its application to particle movements. In the 
third chapter, we start developing the methodological theory of the parametrization of dynamic spatial weight 
matrices in an atmospheric context. The fourth chapter, bring forth the bridge between smooth transition models, 
spatial autoregressive model and our dynamic wight matrices. The fifth chapter describe the observed transport 
process and the hidden emission process we analyse with the model. The sixth chapter presents the empirical 
application of the model by analysing fine pollution particle in the Province of Noord-Holland. Thereafter, a 
discussion of the results and the theoretical connection to the underlying dynamical system follows. At last, we 
close the thesis with a brief conclusion that highlights the main points of interest in the model and the empirical 
study.