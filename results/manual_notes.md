## Manual failure log

### Example 1
Q: Were Scott Derrickson and Ed Wood of the same nationality?
Gold: yes
Model: insufficient information
Sources retrieved: Ed Wood, Woodson Arkansas, Ed Wood (film), Conrad Brooks, Ed Wood (film)
Diagnosis: Retrieval miss -- all 5 retrieved chunks relate to Ed Wood; zero
chunks about Scott Derrickson were retrieved, despite the question requiring
both. Single-shot retrieval with top_k=5 wasn't sufficient here. This is
exactly the failure mode iterative retrieval (Day 5) is meant to address.

### Example 2
Q: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Gold: Chief of Protocol
Supporting facts (per HotpotQA): Kiss and Tell (1945 film), Shirley Temple
Model: insufficient information
Sources retrieved: Kiss and Tell (1945 film), A Kiss for Corliss, Janet Waldo,
Meet Corliss Archer (TV series), A Kiss for Corliss
Diagnosis: Entity disambiguation failure. Two different actresses played
Corliss Archer in different media (Janet Waldo on radio, Shirley Temple in
the 1945 film specifically). Retrieval surfaced the radio actress instead
of the film actress the question actually asks about; "Shirley Temple" was
never retrieved at all despite being one of the 10 provided paragraphs.