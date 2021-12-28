# Boundary-Detection
<h2> Part 2: Ice tracking</h2>

**1. Simplified Model**
For getting the two boundaries using simplified bayes net we used emission probabilities. Emission Probabilities were calculated using edge strength array. Greater the edge strength higher the probability of it being a boundary. We also calculated other list of emission probability such that lower the pixel value for a given cell higher is the probability of it being a boundary. Then we simply returned the row number for each column with higher probability for air-ice boundary. For ice-rock boundary we had precondition that the row number should be atleast 10 pixels below the air-ice boundary. The row with highest emission probability which satisfied this condition was returned as ice-rock boundary for each column. Below are the simplified model results for both air-ice and ice-rock boundary in yellow for test image 30.png. We can see that for most of the columns it identified the correct row numbers for both the boundaries.

![image](https://media.github.iu.edu/user/18146/files/c4d29980-5245-11ec-8bc8-79a7c70f7ab1)
![image](https://media.github.iu.edu/user/18146/files/d0be5b80-5245-11ec-952b-ec8baf7a244c)

**2. HMM**
We further improved the above results by using HMM bayes net. We solved this model using Viterbi algorithm. For calculating the viterbi table we used the previously calculated emission probabilities and calculated transition probabilities such that lesser the distance between the row number of previous column and current column higher is the probability to incorparate the smoothness function. After calculating the whole viterbi table we backtracked to get the sequence of states that is sequence of rows where the boundary is for each column and returned the sequence as air-ice boundary. Then we used the same viterbi algorithm to find the ice-rock boundary just tweaked the values in viterbi table such that probability of all the rows above air-ice boundary + 10 was made 0 such that they never get picked up while backtracking. Below are the results of HMM model for both air-ice and ice-rock boundaries in blue for test image 30.png. We can now see the improvement from above images and can see that the boundaries identidfied are almost perfect!!
 
![image](https://media.github.iu.edu/user/18146/files/33652680-5249-11ec-81be-4a902edc04f9)
![image](https://media.github.iu.edu/user/18146/files/3c55f800-5249-11ec-9aef-5dfc72e4503c)

**3. Human Feedback in HMM**
For incorperating human feedback in our HMM model we just tweaked few of the already calculated probabilities. Transition probabilities were changed a bit such that 10 nearest columns to the column where the human feedback point is present had probability as 1 for the rows closest to the row of human feedback point. This will allow the boundary to pass through that point. Also in viterbi table the coordinate provided by human feedback was given the probability of 1. Below are the results for HMM model when human feedback is provided where the boundaries for both air-ice nad ice-rock are given in red and points in asterisk

![image](https://media.github.iu.edu/user/18146/files/30b80080-524c-11ec-9fdc-ee575164385a)
![image](https://media.github.iu.edu/user/18146/files/3ad9ff00-524c-11ec-96ff-bd4009c2eb24)

Some more sample outputs where the boundaries identified by all 3 models is present in one image respectively for both air-ice and ice-rock (In most of the cases the red line(HMM human feedback) line overlaps the blue line(HMM) as they present almost the same outputs)

![image](https://media.github.iu.edu/user/18146/files/bb98fb00-524c-11ec-8664-b8de140a8c6f)
![image](https://media.github.iu.edu/user/18146/files/c6ec2680-524c-11ec-92c2-4e761b4acd21)

![image](https://media.github.iu.edu/user/18146/files/04e94a80-524d-11ec-9213-1ffd668efbf5)
![image](https://media.github.iu.edu/user/18146/files/0fa3df80-524d-11ec-95de-e49c8f14e644)

<hr>
