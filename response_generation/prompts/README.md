# GPT-4 Prompts

## GINGER components

### Nugget Detection

*"Given a query and a passage, annotate information nuggets that contain the key information answering the query. Copy the text of the passage and put the annotated information nuggets between <IN> and </IN>. Do NOT modify the content of the passage. Do NOT add additional symbols, spaces, etc. to the text. Question: **query** Passage: **passage**"*

### Information Cluster Summarization

*"Summarize the provided information into one sentence (approximately 35 words). Generate one-sentence long summary that is short, concise and only contains the information provided. **Text**"*

### Improving Response Fluency

*"Rephrase the response given a query. Do not change the information included in the response. Do not add information not mentioned in the response. Query: **query** Response: **response**"*

### Response Generation from One Cluster

*"Generate the answer to a query that is 3 sentences long (approximately 100 words in total) using the provided information. Use only the provided information. You can expand the provided information but do not add any additional information. **Text**"*

## Baseline Response Generation

*"Generate the answer to a query that is 3 sentences long (approximately 100 words in total) using the provided information. Use only the provided information and do not add any additional information. Question: **query** Passage: **passages**"*

## Baseline Response Generation (Chain-of-Thought)

TASK

You are an assistant generating responses to user questions based on the provided information. Your response should rely on the context passage and it should not incorporate any additional information.

SPECIFIC STEPS

Follow a structured step-by-step approach to ensure relevance and coherence in the generated response:
- Step 1: extract key pieces of information relevant to the question from the provided passage
- Step 2: group related pieces if information based on the aspect of the topic they discuss or the point of view they represent
- Step 3: rank groups of information based on their relevance to the query
- Step 4: use the top groups of relevant information to write a coherent and concise response

The final response should be three sentences long (approximately 100 words).

EXAMPLE

Question: Tell me more about the Blue Lives Matter movement.

Passage: [passage]The internet facilitates the spread of the message 'All Lives Matter' as a response to the Black Lives Matter hashtag as well as the 'Blue Lives Matter' hashtag as a response to Beyonce's halftime performance speaking out against police brutality.
Following the shooting of two police officers in Ferguson and in response to BLM, the hashtag [[Blue Lives Matter|#BlueLivesMatter]] was created by supporters of the police. Following this, Blue Lives Matter became a pro-police movement in the United States. It expanded after the killings of American police officers.
On December 20, 2014, in the wake of the killings of officers Rafael Ramos and Wenjian Liu, a group of law enforcement officers formed Blue Lives Matter to counter media reports that they perceived to be anti-police. Blue Lives Matter is made up of active and retired law enforcement officers. The current national spokesman for Blue Lives Matter is retired Las Vegas Metropolitan Police Department Lieutenant Randy Sutton.
Originating in New York City in December 2014, Blue Lives Matter NYC is an organization and current nationwide movement that was created to help law enforcement officers and their families during their times of need. Sergeant Joey Imperatrice, along with Officers Chris Brinkley and Carlos Delgado, felt 'compelled to show support for their brothers and sisters in blue' and handed out bracelets that stated 'Blue Lives Matter'. They decided to create an organization, which became the non-profit Blue Lives Matter NYC. This organization's mission is to raise awareness and encourage public aid for the needs of police officers, to help police officers assist one another, and to provide a police officer's family with comfort and support during difficult times. This campaign is designed to raise both awareness and money for families in need. In order to increase nationwide awareness, over three hundred billboards have been posted with the slogan 'Blue Lives Matter'. Many of these billboards are also accompanied by the hashtag #thankublu, which individuals use to show their support for police officers.
Blue Lives Matter is a pro-police movement in the United States. It was started after the killings of NYPD officers Rafael Ramos and Wenjian Liu in Brooklyn, New York, on December 20, 2014, after they were ambushed in their patrol car. Blue Lives Matter was formed in reaction to the Black Lives Matter movement, which seeks to end police brutality against the African American community.[/passage]

Step-by-step response generation process:

Step 1: Extract key pieces of information relevant to the query
'Blue Lives Matter' hashtag as a response to Beyonce's halftime performance speaking out against police brutality
[[Blue Lives Matter|#BlueLivesMatter]] was created by supporters of the police
Blue Lives Matter became a pro-police movement in the United States
a group of law enforcement officers formed Blue Lives Matter to counter media reports that they perceived to be anti-police
made up of active and retired law enforcement officers
help law enforcement officers and their families during their times of need
raise awareness and encourage public aid for the needs of police officers, to help police officers assist one another, and to provide a police officer's family with comfort and support as they go through hard times
raise both awareness and money for the families in need
over three hundred billboards have been posted with the slogan 'Blue Lives Matter'
#thankublu, a hashtag many individuals use to show their support for police officers
It was started after the killings of NYPD officers Rafael Ramos and Wenjian Liu in Brooklyn, New York, on December 20, 2014, after they were ambushed in their patrol car
 which seeks to end police brutality against the African American community

Step 2: Group related pieces of information
Group A – Purpose & Support for Officers:
help law enforcement officers and their families during their times of need
raise awareness and encourage public aid for the needs of police officers, to help police officers assist one another, and to provide a police officer's family with comfort and support as they go through hard times
raise both awareness and money for the families in need
Group B - Origin & Formation:
'Blue Lives Matter became a pro-police movement in the United States
a group of law enforcement officers formed Blue Lives Matter to counter media reports that they perceived to be anti-police
made up of active and retired law enforcement officers
It was started after the killings of NYPD officers Rafael Ramos and Wenjian Liu in Brooklyn, New York, on December 20, 2014, after they were ambushed in their patrol car
which seeks to end police brutality against the African American community
Group C– Broader Context & Media Reaction:
'Blue Lives Matter' hashtag as a response to Beyonce's halftime performance speaking out against police brutality
[[Blue Lives Matter|#BlueLivesMatter]] was created by supporters of the police
#thankublu, a hashtag many individuals use to show their support for police officers
over three hundred billboards have been posted with the slogan 'Blue Lives Matter'

Step 3: Rank groups based on relevance to the query
Group B – Origin & Formation (Most relevant)
Group A – Purpose & Support for Officers
Group C – Broader Context & Media Reaction

Step 4: Generate a coherent, concise response
Response: Blue Lives Matter was founded by active and retired law enforcement officers following the targeted killings of NYPD officers Rafael Ramos and Wenjian Liu on December 20, 2014. It emerged as a pro-police movement aimed at countering what it perceived as anti-police media narratives while supporting officers and their families through fundraising and awareness campaigns. The movement also engages in public outreach through billboards, social media hashtags like #thankublu, and nonprofit initiatives dedicated to aiding law enforcement personnel in times of need.

NOW PERFORM THE TASK ON THE FOLLOWING INPUT