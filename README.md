# Farm to Computer

Our NASA Space Apps Challenge team is monitoring cherry blossoms to chart cherry yields with temperature

## Problem

How do temperature fluctuations affect Michigan cherry season?

### Cherries and Michigan

Not only do cherry trees produce beautiful pink and white blossoms, they serve a major role in the Michigan economy. As the top producer of tart cherries in the US, Michigan cherries are a matter of state pride. The Traverse City annual National Cherry Festival draws in thousands of visitors each year and generates millions of dollars in tourism. Overall cherry production contributes to the hundreds of millions of dollars for in state agriculture industry.

![An image of cherry trees in bloom near Traverse City, MI](https://assets.simpleviewinc.com/sv-traversecity/image/upload/c_fill,h_491,q_100,w_1440/v1/cms_resources/cms_resources/cms_resources/cms_resources/clients/traversecity/1_Cherry_Blossoms_922eab3d-0357-47df-a773-203fade5440b.jpg)

Cherry blossoms (and thus cherry production) require specific temperature conditions to bloom. To produce fruit, cherry blossoms rely on pollinators and a continuation of warm temperatures. In 2012, a sudden freeze killed 90% of Michigan's cherry crops, resulting in over $200 million in losses for Michigan fruit farmers. Rising global temperatures have led to milder winters and temperature fluctuations that can greatly affect cherry production.

## Solution

Our goal is to combine satellite imagery of Michigan cherry orchards with weather data to understand how cherry blossom timing and yields are affected by temperature patterns.

### Identifying Blossoms

A recent study by [Chen, Jin, & Brown (2019)](https://www.sciencedirect.com/science/article/pii/S092427161930190X) demonstrated that satellite imagery from Sentinel-2 and PlanetScope can be used to identify flowering events for almond orchards. Almond trees produce pink-white flowers similar to cherry trees. These types of blooms are highly reflective. Chen et al. developed the Enhanced Bloom Index (EBI) to identify the onset of blooming:

$$ {\rm EBI} = \frac{{\rm R} + {\rm G} + {\rm B}}{\frac{{\rm G}}{B} \left( {\rm R} - {\rm B} + \epsilon \right)} $$

where $\epsilon$ is the normalizing factor for the scale values being used (e.g., $\epsilon = 256$ for RGB data, which takes on a value between 0 and 255, or $\epsilon = 1$ for reflectance data, which takes on a value between 0 and 1).

They also demonstrated that the EBI was linearly correlated with Bloom Coverage (BC) identified through high resolution (2.5-5.2 cm) aerial imaging. BC corresponds to the fraction of all image pixels containing a flower. This means that the EBI can also be used to quantify the abundance of flower blooms in an image.

### Charting Weather: Conditions for blossoming

Cherry trees rely on two crucial processes in order to bloom. 

#### Chill hours

Cherry trees become dormant during the winter time, when temperatures drop below freezing. To start waking up, cherry trees require a certain number of "chill hours", that is temperatures in the range 32-45 F. The majority of cherries grown in Michigan are tart cherries, which require 1200 or more chill hours to produce blossoms.

During the chill phase, warm days can lead to set-backs. If temperatures rise above 60 F, the chill hour "clock" will wind back or get reset entirely. This process is highly complicated, and there are few models one can use to track chill hours:
- The **simplest model** tracks the total number of hours in the temperature range 32-45 F. It does not account for set backs due to unusually warm days.
- The **Utah model** adds negative time for hours above 60 F during the chill hours accumulation phase.
- The **dynamic model** considers a two step process. First, a temperature-sensitive compound accumulates during chill hours, and can be destroyed by warmer temperatures (the reversible process). When enough of the precursor compound accumulates, it can be converted to a more stable substance that is stored and does not get destroyed (the irreversible process).

Michigan experiences temperature swings that require more complex models to fully understand and predict how the cherry season will proceed.

#### Ecodormancy and growing degree days (GDD)

Once a tree has met its "chill hour" requirement, it enters ecodormacy. At this point, it requires warmer temperatures to resume growth. The "growing degree days" (GDD) metric help us determine when the blooming events are going to occur.

GDD is calculated by multiplying the number of F degrees above 41 F by the number of days at that temperature. Mathematically, for each day $i$ after the chill hour criteria is met:

$$ {\rm GDD} = \Sigma_i (T_i - 41\ {\rm F}) $$

where $T_i$ is the temperature on day $i$.

Montmorency variety tart cherries (accounting for the majority of Michigan tart cherry production) bloom around 230-250 GDD.

#### Risks from temperature fluctuations

Polinated cherry blossoms turn into cherry fruits that can be harvested in the summer. If the temperature rises too quickly after a cherry bloom event, the flowers can wilt; if there is a sudden freeze, the flowers can be damaged and fall off. The destruction of cherry flowers prevents cherry fruits from forming, so these events like these can dramatically affect cherry yields across the state.

### Datasets

For our prototype, we decided to focus on the area around [Shoreline Fruits](https://www.shorelinefruit.com/about/our-story), which is the largest cherry orchard in Michigan (60 acres). It is located near Traverse City. We scanned news articles to identify the dates of the most recent cherry blossom, which started on April 29, 2025.

We downloaded datasets covering Shoreline Fruits from [Sentinel-2](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2) and [PlanetScope](https://earth.esa.int/eogateway/missions/planetscope) databases. 

We are using data from national weather stations to track daily temperatures. We used the `meteostat` Python package to gather hourly temperature information from Cherry Capital Airport in Traverse City and the Antrim County Airport, which are the two nearest airports to Shoreline Fruits. Then we used `geopy` to interpolate this data to get the temperature at the orchard.

## Impact

What can we expect in the future?

## What's next?

What can apple farmers do to prevent crop losses?

## AI Statement

We used Google Gemini for help researching the blooming conditions for cherry trees, as well as the cherry's significance to Michigan agriculture. This is how we learned about chill hours and GDD. All writing in this document was written independently after synthesizing the necessary information.


