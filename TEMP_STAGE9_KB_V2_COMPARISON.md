# Temporary V1/V2/V3 KB Rule-RAG comparison

This preview keeps the same 20 locked cases and the same `gemma4:12b` generator 
as `TEMP_STAGE9_PAIRED_EXPLANATION_REVIEW.md`. It does not alter the frozen KB, 
Stage 7/8/9 outputs, or the final evaluation. V2 traces are retrieved in memory 
with the same strict antecedent gate and the locked candidate preserved.

The V2 file marks all rules `rebuilt_v2_substantive_expert_rule`; that label is 
accepted only for this isolated preview instead of the frozen production `retain` label.

After this preview, K047's unsupported `leather skirt` candidate term was removed during
the bounded V2 validation. Consequently, example 06 no longer has a valid V2 trace and
its displayed V2 response is retained only as a pre-correction preview, not as valid evidence.

## 01 — bags — case-051b60b8ffa574ab

- Request: Recommend a bag that completes this outfit.
- Query item: zippers faux leather ponte pants
- Locked recommendation: nylon cross body bag
- Original V1 trace: K070
- V2 exact trace: K072
- No-RAG (existing): The nylon cross body bag provides a practical contrast to the texture of the zippers faux leather ponte pants.
- V1 Rule-RAG (existing): To complete your outfit, I recommend the nylon cross body bag. A shoulder bag, tote, or clutch is a documented finishing direction when trousers, pants, or shorts are present [K070].
- V2 Rule-RAG: To complete your look, I recommend the nylon cross body bag. Because your pants are a statement-making faux leather material, a clean crossbody bag provides a balanced contrast that keeps the overall look from feeling overloaded [K072].
- V3 exact trace: K161
- V3 Rule-RAG: The nylon cross body bag is an excellent choice to complete your look. Because your ponte pants offer a streamlined silhouette, a practical crossbody provides a balanced contrast to the structured texture of the fabric. This selection ensures the ensemble remains functional while maintaining a cohesive aesthetic that complements the specific proportions of your trousers. [K161]

## 02 — bags — case-0c8fe6d54112ca26

- Request: Recommend a bag that completes this outfit.
- Query item: flat
- Locked recommendation: cut n
- Original V1 trace: K074, K070, K037
- V2 exact trace: NO STRICTLY APPLICABLE RULE
- No-RAG (existing): The cut n. complements your look by providing a balanced silhouette to your flat.
- V1 Rule-RAG (existing): I recommend the cut n to complete your outfit. When flat shoes are explicit, a shoulder bag or basket bag is a documented finishing direction [K074].
- V2 Rule-RAG: NOT GENERATED: no V2 rule passed the strict antecedent gate for this locked case, so a Rule-RAG response would be unsupported.
- V3 exact trace: NO STRICTLY APPLICABLE RULE
- V3 Rule-RAG: NOT GENERATED: no V3 rule passed the strict antecedent gate for this locked case, so a Rule-RAG response would be unsupported.

## 03 — bags — case-0cac5b9e1c55b529

- Request: Recommend a bag that completes this outfit.
- Query item: sam edelman contrast leather platform pump
- Locked recommendation: buckles pu leather metal shoulder bag
- Original V1 trace: K073, K096, K036, K031
- V2 exact trace: K075
- No-RAG (existing): The buckles pu leather metal shoulder bag complements the height and structure of the sam edelman contrast leather platform pump.
- V1 Rule-RAG (existing): To complete your outfit, I recommend the buckles pu leather metal shoulder bag. A shoulder bag is a documented finishing direction when pumps are explicit [K073].
- V2 Rule-RAG: The buckles pu leather metal shoulder bag is a great choice because shoulder bags and structured top handles work naturally with pumps [K075].
- V3 exact trace: K075, K182
- V3 Rule-RAG: The buckles pu leather metal shoulder bag is an excellent choice to complete your look. Because your pumps feature a polished finish, this structured shoulder bag maintains a sophisticated aesthetic by matching the level of polish rather than requiring an exact color match [K182]. The inclusion of metal hardware creates a cohesive connection with the footwear, ensuring the accessories complement each other without competing for attention [K075].

## 04 — bags — case-0cc5d527ad1c2086

- Request: Recommend a bag that completes this outfit.
- Query item: georgia ankle boot women's boots
- Locked recommendation: crossbody bag bags
- Original V1 trace: K071, K029, K070
- V2 exact trace: K096
- No-RAG (existing): A crossbody bag bags provides a practical and hands-free way to complete your look while complementing the silhouette of the georgia ankle boot women's boots.
- V1 Rule-RAG (existing): I recommend the crossbody bag bags to complete your look. A shoulder bag or tote is a documented finishing direction when ankle booties or booties are explicit [K071].
- V2 Rule-RAG: To complete your look, I recommend the crossbody bag bags. Boots support bags with a little visual substance, including crossbodies [K096].
- V3 exact trace: K181, K096
- V3 Rule-RAG: The crossbody bag bags is an excellent choice to complete your look. Because your georgia ankle boot women's boots provide a sleek silhouette, they pair well with a bag that offers a balanced amount of visual substance. This selection ensures the accessory complements the boot's presence without creating a disconnected look, maintaining a cohesive aesthetic for your ensemble. [K181] [K096]

## 05 — bottoms — case-00394c0d34524ad6

- Request: Recommend bottoms that work with this outfit.
- Query item: solid homme cotton t-shirt
- Locked recommendation: women's casual elastic waist pants
- Original V1 trace: K101
- V2 exact trace: K008
- No-RAG (existing): The women's casual elastic waist pants provide a comfortable and versatile pairing for the solid homme cotton t-shirt.
- V1 Rule-RAG (existing): You can pair your top with women's casual elastic waist pants to complete an everyday outfit. [K101]
- V2 Rule-RAG: You can pair your solid homme cotton t-shirt with women's casual elastic waist pants. A t-shirt can move from very casual to polished depending on the bottom [K008].
- V3 exact trace: K008, K134
- V3 Rule-RAG: The women's casual elastic waist pants are an excellent choice to complement your solid homme cotton t-shirt. This pairing works because elastic-waist trousers provide a soft, comfortable silhouette that balances the simplicity of a basic tee for everyday wear. By choosing a defined waistline, you ensure the outfit feels like a cohesive ensemble rather than a collection of unrelated basics. [K008]

## 06 — bottoms — case-014db2eb6705d417

- Request: Recommend bottoms that work with this outfit.
- Query item: giuseppe zanotti women's studded boot
- Locked recommendation: balmain leather skirt chains
- Original V1 trace: K110
- V2 exact trace: K047
- No-RAG (existing): I recommend the balmain leather skirt chains to complement the texture and aesthetic of the giuseppe zanotti women's studded boot.
- V1 Rule-RAG (existing): The balmain leather skirt chains are a suitable choice because a bottom-category item can provide the lower-body component of an outfit formula when the query item is shoes [K110].
- V2 Rule-RAG: You should pair your outfit with balmain leather skirt chains. Since boots can change the balance of the lower half, choosing a skirt provides a deliberate integration with the footwear [K047].
- V3 exact trace: K213
- V3 Rule-RAG: The balmain leather skirt chains are an excellent choice to complement your footwear. Because both the boots and the skirt feature hardware, the leather texture creates a cohesive, edgy aesthetic that feels intentional rather than busy. This pairing works effectively because the shared metallic elements connect the two pieces, while the simplicity of the skirt provides necessary visual relief to balance the statement nature of the studded boots. [K213]

## 07 — bottoms — case-016755b4f2bd3327

- Request: Recommend bottoms that work with this outfit.
- Query item: boho fringed shoulder bag celebrity fashion lookbook
- Locked recommendation: mesh panel skirt
- Original V1 trace: K118
- V2 exact trace: K118
- No-RAG (existing): The mesh panel skirt complements the bohemian aesthetic of the boho fringed shoulder bag.
- V1 Rule-RAG (existing): The mesh panel skirt is a suitable choice because a bottom-category item can provide the documented lower-body component of an outfit formula when the query item is a bag [K118].
- V2 Rule-RAG: You should pair your look with the mesh panel skirt. Since the bag is a focal point, a solid skirt provides a balanced contrast to the accessory [K118].
- V3 exact trace: K189
- V3 Rule-RAG: The mesh panel skirt is an excellent choice to complement your boho fringed shoulder bag. Because the bag features prominent fringe, choosing a simpler lower half ensures the accessory remains the primary focal point of your ensemble. This balance allows the intricate textures of the bag to stand out while maintaining a cohesive, sophisticated silhouette throughout the entire look. [K189]

## 08 — bottoms — case-034821767e00ee16

- Request: Recommend bottoms that work with this outfit.
- Query item: louis vuitton monogram satchel
- Locked recommendation: product shopping
- Original V1 trace: K118
- V2 exact trace: NO STRICTLY APPLICABLE RULE
- No-RAG (existing): I recommend product shopping to find bottoms that complement your louis vuitton monogram satchel.
- V1 Rule-RAG (existing): I recommend product shopping to complete your look. A bottom-category item provides the documented lower-body component of an outfit formula when the query item is a bag [K118].
- V2 Rule-RAG: NOT GENERATED: no V2 rule passed the strict antecedent gate for this locked case, so a Rule-RAG response would be unsupported.
- V3 exact trace: NO STRICTLY APPLICABLE RULE
- V3 Rule-RAG: NOT GENERATED: no V3 rule passed the strict antecedent gate for this locked case, so a Rule-RAG response would be unsupported.

## 09 — outerwear — case-0173bdb2934d615e

- Request: Recommend outerwear that works with this outfit.
- Query item: black chunky heel point toe pu boots
- Locked recommendation: black single-breasted loose fit coat
- Original V1 trace: K111
- V2 exact trace: K091, K069
- No-RAG (existing): The black single-breasted loose fit coat provides a balanced silhouette that complements the proportions of the black chunky heel point toe pu boots.
- V1 Rule-RAG (existing): The black single-breasted loose fit coat provides a complete outfit formula when paired with your footwear [K111].
- V2 Rule-RAG: You should wear the black single-breasted loose fit coat. Because your chunky boots are a substantial style, they pair well with substantial coats [K091].
- V3 exact trace: K091, K069
- V3 Rule-RAG: The black single-breasted loose fit coat is an excellent choice to complement your footwear. Because your boots feature a chunky silhouette, they pair naturally with substantial coats to create a cohesive, grounded look [K091]. This specific selection provides a sophisticated balance by offering a clean, structured line that complements the polished aesthetic of your pointed-toe heels without competing with the bold texture of the boots [K069].

## 10 — outerwear — case-0216651fe59b29df

- Request: Recommend outerwear that works with this outfit.
- Query item: mini skirt
- Locked recommendation: chicnova fashion split zipper
- Original V1 trace: K107
- V2 exact trace: K064
- No-RAG (existing): The chicnova fashion split zipper provides a structured layer that complements the proportions of a mini skirt.
- V1 Rule-RAG (existing): The chicnova fashion split zipper is a suitable choice because an outerwear-category item can provide the necessary layer for an outfit featuring bottoms [K107].
- V2 Rule-RAG: I recommend the chicnova fashion split zipper to complement your mini skirt. Longer coats can provide useful proportion and seasonal balance when paired with mini skirts [K064].
- V3 exact trace: K064
- V3 Rule-RAG: The chicnova fashion split zipper provides a sophisticated layer that balances the proportions of your mini skirt. By incorporating a structured outer layer, you can transition the look from a casual aesthetic to a more polished and intentional ensemble. This specific piece offers the necessary coverage to create a balanced silhouette while maintaining the original style of the skirt. [K064]

## 11 — outerwear — case-03ef2447e81a3c05

- Request: Recommend outerwear that works with this outfit.
- Query item: classic flap bag
- Locked recommendation: barbara bui blazer
- Original V1 trace: K120
- V2 exact trace: K103
- No-RAG (existing): The barbara bui blazer provides a structured silhouette that complements the sophisticated aesthetic of the classic flap bag.
- V1 Rule-RAG (existing): The barbara bui blazer provides a structured layer to complete your look. [K120]
- V2 Rule-RAG: The barbara bui blazer provides a polished look that pairs well with your bag. [K103]
- V3 exact trace: K103, K197
- V3 Rule-RAG: The barbara bui blazer is an excellent choice to complement your classic flap bag. This pairing works because a blazer provides a sophisticated city polish that aligns perfectly with the structured aesthetic of a flap bag. By choosing a tailored silhouette, you ensure the bag remains a focal point while maintaining a polished look. This combination creates a cohesive, intentional ensemble that balances professional style with timeless accessory appeal. [K103] [K197]

## 12 — outerwear — case-08579e6c896be310

- Request: Recommend outerwear that works with this outfit.
- Query item: laser cut saddle bag
- Locked recommendation: cheap womens outerwear buy winter women online
- Original V1 trace: K120
- V2 exact trace: K103
- No-RAG (existing): I recommend cheap womens outerwear buy winter women online to complement the laser cut saddle bag.
- V1 Rule-RAG (existing): You can pair your laser cut saddle bag with cheap womens outerwear buy winter women online to provide the documented layer in your outfit formula [K120].
- V2 Rule-RAG: You can pair your laser cut saddle bag with cheap womens outerwear buy winter women online. This style of bag works well with various types of outerwear, such as a blazer, trench, leather jacket, denim jacket, or tailored coat [K103].
- V3 exact trace: K197, K103
- V3 Rule-RAG: I recommend the cheap womens outerwear buy winter women online to complement your look. This choice works because a saddle bag pairs well with various outerwear styles, such as tailored coats or leather jackets, to create a balanced aesthetic. Ensuring the strap sits comfortably over the shoulder without catching on bulky sleeves ensures the ensemble looks intentional and polished rather than cluttered. [K197] [K103]

## 13 — shoes — case-02eee18676b73841

- Request: Recommend shoes that complete this outfit.
- Query item: stylish scoop neck loose emoji print crop top women
- Locked recommendation: topshop vintage trainers
- Original V1 trace: K102
- V2 exact trace: K016
- No-RAG (existing): The topshop vintage trainers provide a balanced contrast to the relaxed silhouette of the crop top.
- V1 Rule-RAG (existing): The topshop vintage trainers complete your outfit as a suitable footwear option for your top [K102].
- V2 Rule-RAG: The topshop vintage trainers are a great choice because sneakers are suitable for daytime and casual wear when pairing a crop top [K016].
- V3 exact trace: K016
- V3 Rule-RAG: The topshop vintage trainers provide a grounded, effortless finish to your look. Because a crop top can sometimes feel very bold, choosing a classic sneaker helps balance the proportions and keeps the overall aesthetic grounded for a casual daytime vibe. This choice ensures the footwear complements the relaxed nature of the top without over-styling the ensemble. [K016]

## 14 — shoes — case-03c159181e5dee57

- Request: Recommend shoes that complete this outfit.
- Query item: river island pink tie dye la print tank top
- Locked recommendation: paolo iantorno candy pearl pink lace platform pumps
- Original V1 trace: K102
- V2 exact trace: K019
- No-RAG (existing): The paolo iantorno candy pearl pink lace platform pumps complement the soft tones of the tank top while adding a feminine texture to the look.
- V1 Rule-RAG (existing): The paolo iantorno candy pearl pink lace platform pumps complete the outfit as footwear. [K102]
- V2 Rule-RAG: To complete your look, I recommend the paolo iantorno candy pearl pink lace platform pumps. Since a tank top creates a visually light outfit, choosing footwear with texture or a strong shape adds interest to the minimal top [K019].
- V3 exact trace: K019
- V3 Rule-RAG: The paolo iantorno candy pearl pink lace platform pumps are an excellent choice to complete your look. Because a tank top creates a visually light aesthetic, these shoes provide the necessary structure and texture to define the overall style. The intricate lace and elevated heel add a sophisticated touch that complements the minimal nature of the top while providing a clear, intentional finish to the ensemble. [K019]

## 15 — shoes — case-043797c4cb7dee57

- Request: Recommend shoes that complete this outfit.
- Query item: off-white off-white metallic bomber jacket silver
- Locked recommendation: ash open toe wedge sneakers silver
- Original V1 trace: K115
- V2 exact trace: K060
- No-RAG (existing): The ash open toe wedge sneakers silver complement the metallic finish of the jacket to create a cohesive, cohesive look.
- V1 Rule-RAG (existing): To complete your look, I recommend the ash open toe wedge sneakers silver. Since your outerwear piece requires a footwear component, these sneakers provide a suitable option for your outfit [K115].
- V2 Rule-RAG: To complete your look, I recommend the ash open toe wedge sneakers silver. These sneakers provide a casual finish to your bomber jacket [K060].
- V3 exact trace: K060
- V3 Rule-RAG: The ash open toe wedge sneakers silver provide a sophisticated way to complete your look. By choosing sneakers, you successfully transition the metallic bomber jacket into a more casual aesthetic while maintaining a polished feel. This choice balances the structured silhouette of the outerwear with a modern, comfortable finish that complements the metallic tones of the ensemble. [K060]

## 16 — shoes — case-049f4c3321d95a46

- Request: Recommend shoes that complete this outfit.
- Query item: new york bella tote star style
- Locked recommendation: dolce women's shoes
- Original V1 trace: K119
- V2 exact trace: K089
- No-RAG (existing): You should pair your look with dolce women's shoes to complete the outfit.
- V1 Rule-RAG (existing): To complete your outfit, I recommend dolce women's shoes. A shoe-category item provides the documented footwear component of an outfit formula when the query item is a bag [K119].
- V2 Rule-RAG: To complete your look, I recommend dolce women's shoes. A structured work or city bag pairs best with shoes that look equally intentional [K089].
- V3 exact trace: K194, K089
- V3 Rule-RAG: The dolce women's shoes provide a polished finish that complements the structured aesthetic of your bag. Because a structured work tote benefits from cleaner and more polished footwear, these shoes create the necessary visual coherence for a professional look. This pairing ensures that both the bag and footwear appear equally intentional without being overly decorative, maintaining a sophisticated balance for your ensemble. [K194] [K089]

## 17 — tops — case-026665827bfaaf65

- Request: Recommend a top that works with this outfit.
- Query item: givenchy skirt green leopard-print stretch-jersey
- Locked recommendation: shein sheinside green puff sleeve split chiffon blouse
- Original V1 trace: K105
- V2 exact trace: K004
- No-RAG (existing): The shein sheinside green puff sleeve split chiffon blouse complements the skirt by coordinating with the green color palette.
- V1 Rule-RAG (existing): The shein sheinside green puff sleeve split chiffon blouse is a suitable choice because a top-category item provides the documented upper-body component of an everyday outfit formula when the query item is bottoms [K105].
- V2 Rule-RAG: I recommend the shein sheinside green puff sleeve split chiffon blouse. Because a skirt's shape and texture should determine the top, a blouse provides a polished look that balances the visual elements of the outfit [K004].
- V3 exact trace: K004
- V3 Rule-RAG: The shein sheinside green puff sleeve split chiffon blouse is an excellent choice for your look. By selecting a blouse, you elevate the ensemble to a more polished aesthetic while maintaining a cohesive color story. This choice works because it balances the bold texture of the leopard print by keeping the overall silhouette intentional and sophisticated rather than visually busy. [K004]

## 18 — tops — case-094db2a4cbd25ec7

- Request: Recommend a top that works with this outfit.
- Query item: nike printed leggings
- Locked recommendation: urban classics ladies tank top button top
- Original V1 trace: K105
- V2 exact trace: K003
- No-RAG (existing): The urban classics ladies tank top button top provides a structured contrast to the casual aesthetic of the nike printed leggings.
- V1 Rule-RAG (existing): The urban classics ladies tank top button top provides the upper-body component of an everyday outfit formula when paired with bottoms [K105].
- V2 Rule-RAG: To complement your nike printed leggings, I recommend the urban classics ladies tank top button top. Since leggings are a close-fitting bottom, a fitted tank can work for an athleisure look when balanced with other elements [K003].
- V3 exact trace: K003
- V3 Rule-RAG: The urban classics ladies tank top button top provides a structured contrast to the sleek silhouette of your leggings. By pairing a fitted bottom with a more structured top, you create a balanced proportion that elevates the look beyond basic gym wear. This combination effectively transitions the outfit into a more polished aesthetic while maintaining a cohesive, intentional style. [K003]

## 19 — tops — case-09628ef84cbf20ab

- Request: Recommend a top that works with this outfit.
- Query item: topshop side curve leggings
- Locked recommendation: basic peplum top
- Original V1 trace: K105
- V2 exact trace: K003
- No-RAG (existing): The basic peplum top provides a structured silhouette that complements the form-fitting nature of the topshop side curve leggings.
- V1 Rule-RAG (existing): I recommend the basic peplum top. This top-category item provides the documented upper-body component of an everyday outfit formula when the query item is bottoms [K105].
- V2 Rule-RAG: I recommend the basic peplum top. Because you are wearing leggings, which are a close-fitting bottom, choosing a more structured piece like a peplum top creates a clearer proportion than making the entire outfit equally tight [K003].
- V3 exact trace: K003
- V3 Rule-RAG: The basic peplum top is an excellent choice to pair with your leggings. Because leggings are very close-fitting, incorporating a more structured silhouette like a peplum creates a more intentional and balanced proportion for your look. This contrast between the fitted bottom and the structured top provides a more polished aesthetic than wearing two tight pieces together. [K003]

## 20 — tops — case-0b8aa30b40b7cd27

- Request: Recommend a top that works with this outfit.
- Query item: boohoo alice crochet lace tailored shorts
- Locked recommendation: alice olivia cropped top
- Original V1 trace: K105
- V2 exact trace: K005
- No-RAG (existing): The alice olivia cropped top complements the texture of the boohoo alice crochet lace tailored shorts by balancing the intricate lace with a structured silhouette.
- V1 Rule-RAG (existing): The alice olivia cropped top is a suitable choice because a top-category item provides the documented upper-body component of an everyday outfit formula when the query item is bottoms [K105].
- V2 Rule-RAG: The alice olivia cropped top is a great choice because a tucked waist or cropped silhouette can prevent tailored shorts from looking shapeless [K005].
- V3 exact trace: K005
- V3 Rule-RAG: The alice olivia cropped top is an excellent choice to complement your tailored shorts. By utilizing a tucked waist or a structured silhouette, this top ensures the ensemble maintains a polished, smart-casual aesthetic rather than appearing shapeless. This pairing effectively balances the proportions of the shorts while maintaining a cohesive look that works perfectly for a sophisticated daytime outing. [K005]
