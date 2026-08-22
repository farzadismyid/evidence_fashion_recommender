# Stage 12 frozen qualitative examples

All records were selected deterministically from frozen canonical outputs; text and verifier results are unedited.

## Exact-Trace strong paired

- Case / generator / condition: `case-b84372a3927144c5` / `llama3.1:8b-instruct-q8_0` / `rule_rag`
- Target category; trace size: tops; 2
- Explanation: For a cohesive look with the Vince Camuto tapered strap high heel sandal, consider the Vero Moda halter top. This choice works because it adheres to the guideline of keeping the upper half seasonally light [K041], making it suitable for holiday and daytime wear. The halter top's simplicity also allows the refined sandal to take center stage, avoiding excessive embellishment [K039].
- Extracted claim: The Vince Camuto tapered strap high heel sandal and the Vero Moda halter top form a cohesive look.
- Evidence/reference: [K041] With sandals, wedges, espadrilles, or clogs, keep the upper half seasonally light. Linen or cotton shirts, tanks, wrap tops, simple tees, and light blouses suit holiday and daytime wear; a silk camisole or refined sleeveless top works for a summer dinner. Natural textures and lighter colours usually make the outfit feel coherent, while a more structured top can stop platform or chunky sandals from reading too beachy.; [K039] Pumps and low heels can either sharpen a casual outfit or support a dressier one, so the top should decide which direction you want. Pair them with a fine knit, crisp shirt, or minimal tee for daytime polish; use a silk blouse, camisole, or elegant fitted top for dinner or events. Because the shoes already add refinement, the top does not need excessive embellishment unless the rest of the outfit is deliberately evening-focused.
- Verifier outcome: exact=supported; full_kb=supported; common=N/A
- Selection / interpretation: deterministic 75%-quantile paired exact_delta

## Exact-Trace weak paired

- Case / generator / condition: `case-f9d36728095e2ded` / `llama3.1:8b-instruct-q8_0` / `no_rag`
- Target category; trace size: shoes; 1
- Explanation: Tabitha Simmons Tabitha Simmons pumps would be a great addition to the outfit with Nicole white trousers. The sleek design and neutral color of the pumps will complement the clean lines and crisp white of the trousers, creating a polished and put-together look. The pumps' subtle elegance will also add a touch of sophistication to the overall ensemble, making it suitable for a variety of occasions.
- Extracted claim: Tabitha Simmons pumps would be a great addition to the outfit with Nicole white trousers.
- Evidence/reference: {"locked_item_minimal_name": "tabitha simmons tabitha simmons pumps", "query_item_minimal_name": "nicole white trousers", "user_request": "Recommend shoes that complete this outfit."}
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=N/A
- Selection / interpretation: deterministic 25%-quantile paired exact_delta

## Full-KB strong paired

- Case / generator / condition: `case-08579e6c896be310` / `gemma4:12b` / `rule_rag`
- Target category; trace size: outerwear; 2
- Explanation: I recommend the cheap womens outerwear buy winter women online to complement your look. This choice works because a saddle bag pairs well with various outerwear styles, such as tailored coats or leather jackets, to create a balanced aesthetic. Ensuring the strap sits comfortably over the shoulder without catching on bulky sleeves ensures the ensemble looks intentional and polished rather than cluttered. [K197] [K103]
- Extracted claim: The saddle bag pairs well with tailored coats.
- Evidence/reference: [K197] Shoulder and saddle bags work with most everyday outerwear, but strap clearance and bag structure matter. Blazers, trenches, leather jackets, denim jackets, and tailored coats all work; use a structured bag to sharpen softer outerwear and a relaxed saddle shape to soften tailoring. Check that the strap sits comfortably over the shoulder and does not fight bulky sleeves or large lapels, because practical fit is part of whether the combination looks intentional.
- Verifier outcome: exact=supported; full_kb=supported; common=N/A
- Selection / interpretation: deterministic 75%-quantile paired full_delta

## Full-KB weak paired

- Case / generator / condition: `case-04c0e19c3be2d124` / `llama3.1:8b-instruct-q8_0` / `no_rag`
- Target category; trace size: shoes; 1
- Explanation: To complete this outfit, I recommend pairing the oversized cardigan with play sneakers. The relaxed fit and casual vibe of the play sneakers will balance out the loose, comfortable silhouette of the cardigan, creating a laid-back and effortless look. This combination will add a sporty touch to the overall ensemble, perfect for a casual day out or a weekend brunch.
- Extracted claim: The oversized cardigan should be paired with play sneakers.
- Evidence/reference: [K018] Knitwear works with both soft and structured shoes, so use texture and silhouette to choose. Sneakers or chunky boots suit relaxed sweaters, loafers and ballet flats sharpen fine knits, and ankle boots or pumps can move a fitted knit toward work or dinner. When the knit is very heavy, a shoe with some visual substance prevents the lower half from looking too delicate; with fine-gauge knitwear, cleaner low-profile shoes usually feel more balanced.
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=N/A
- Selection / interpretation: deterministic 25%-quantile paired full_delta

## Density high paired

- Case / generator / condition: `case-e3e8650b6815c408` / `llama3.1:8b-instruct-q8_0` / `rule_rag`
- Target category; trace size: shoes; 1
- Explanation: For this plus size white step hem t-shirt, I recommend the mother suede sneakers. These shoes will keep the outfit relaxed and casual, while the T-shirt remains a neutral base that can adapt to different contexts. By choosing sneakers, we're setting the outfit's level of polish to a laid-back tone, allowing the rest of the look to evolve freely. [K015]
- Extracted claim: The item is a plus size white step hem t-shirt.
- Evidence/reference: [K015] A T-shirt can support almost any casual shoe, so choose footwear to set the outfit's level of polish. Sneakers keep it relaxed, loafers or ballet flats make it cleaner, ankle boots add weight for cooler weather, and pumps or kitten heels can deliberately dress up a simple tee. The shoe should usually be the stronger formality signal, while the T-shirt remains the neutral base that lets the rest of the outfit change context.
- Verifier outcome: exact=supported; full_kb=supported; common=supported
- Selection / interpretation: deterministic 75%-quantile paired density_delta

## Density low paired

- Case / generator / condition: `case-a7ca4398c1c6e424` / `ministral-3:14b-instruct-2512-q4_K_M` / `no_rag`
- Target category; trace size: bottoms; 2
- Explanation: **"Vila skinny fit jeans"** pair seamlessly with spike heel sandals by balancing structure and fluidity. The snug, streamlined silhouette complements the elongated lines of the heels without competing for attention, creating a cohesive, elongated look. The minimalist fit ensures movement remains unrestricted while maintaining a polished, cohesive aesthetic.
- Extracted claim: "Vila skinny fit jeans" pair seamlessly with spike heel sandals.
- Evidence/reference: [K048] Pumps and heels can either sharpen denim or complete a dressier bottom. Tailored or wide-leg trousers work well when the hem is long enough to skim rather than hide the shoe; straight jeans become more polished with a pointed or kitten heel; midi and mini skirts suit work, dinner, or events depending on fabric. Because the heel adds refinement, the bottom can stay comparatively simple and still look dressed.
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=N/A
- Selection / interpretation: deterministic 25%-quantile paired density_delta

## UIFR strong factual

- Case / generator / condition: `case-6c744bf50cc7c913` / `ministral-3:14b-instruct-2512-q4_K_M` / `no_rag`
- Target category; trace size: bags; 3
- Explanation: The **bucket crossbody bag** complements the **aldo aldo high heel peep toe shoe boot** by balancing its sleek, elongated silhouette with a structured yet versatile accessory. Its compact, crossbody design ensures effortless hands-free functionality while maintaining a polished look that aligns with the boot’s refined aesthetic. The practicality of the bag’s form enhances the outfit’s cohesion without overpowering its minimalist appeal.
- Extracted claim: The bucket crossbody bag is a crossbody bag.
- Evidence/reference: {"locked_item_minimal_name": "bucket crossbody bag", "query_item_minimal_name": "aldo aldo high heel peep toe shoe boot", "user_request": "Recommend a bag that completes this outfit."}
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=supported
- Selection / interpretation: deterministic median exact-support record

## UIFR weak factual

- Case / generator / condition: `case-56ddc70400ea430f` / `gemma4:12b` / `rule_rag`
- Target category; trace size: tops; 1
- Explanation: The stylish one-shoulder solid color top women is an excellent choice to pair with your cotton skirt. By choosing a solid color, you ensure the outfit remains visually balanced and sophisticated rather than overly busy [K004]. This specific silhouette provides a deliberate contrast to the skirt's texture, creating a polished look that highlights the intentionality of the overall ensemble [K004].
- Extracted claim: The top is one-shoulder.
- Evidence/reference: {"locked_item_minimal_name": "stylish one-shoulder solid color top women", "query_item_minimal_name": "cotton skirt", "user_request": "Recommend a top that works with this outfit."}
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=not_supported
- Selection / interpretation: deterministic median exact-support record

## Citation entails

- Case / generator / condition: `case-e7127ff79fe12b03` / `gemma4:12b` / `rule_rag`
- Target category; trace size: tops; 2
- Explanation: The denim swing crop top is an excellent choice to pair with your new block denim shorts. By selecting a top that provides a defined silhouette, you ensure the outfit remains intentional rather than looking shapeless [K005]. This specific pairing creates a balanced, cohesive look by utilizing a lighter top to contrast with the weight of the denim [K001].
- Extracted claim: A top that provides a defined silhouette ensures the outfit remains intentional rather than looking shapeless.
- Evidence/reference: [K005] For shorts, especially in warm weather, use the top to control formality. Linen or cotton shirts, polos, tanks, and clean tees keep beach, holiday, and daytime outfits breathable; tailored Bermuda shorts can take a tucked blouse, ribbed knit, or crisp button-down for brunch or smart-casual use. Light colours and natural fabrics keep the outfit seasonally coherent, while a sharper collar or tucked waist stops longer shorts from looking shapeless.
- Verifier outcome: exact=supported; full_kb=supported; common=N/A
- Selection / interpretation: deterministic median exact-support record

## Citation does not entail

- Case / generator / condition: `case-8633eb7ac4daa293` / `gemma4:12b` / `rule_rag`
- Target category; trace size: tops; 2
- Explanation: The vero moda halter top is an excellent choice to complement your leather crossbody bag. Because a structured leather bag pairs well with halter tops for a polished city look, this combination creates a sophisticated aesthetic. The clean lines of the halter neck ensure the bag remains a clear focal point without competing with a busy neckline. [K185] [K080]
- Extracted claim: The vero moda halter top is an excellent choice to complement your leather crossbody bag.
- Evidence/reference: [K185] Shoulder and saddle bags can work with many tops, so use the bag's structure to set the tone. A leather or structured shoulder bag pairs well with blouses, fitted knits, button-ups, halter tops, and neat tees for city or dinner; a softer saddle bag works naturally with casual shirts, sweaters, and relaxed tops. If the bag has prominent hardware or fringe, keep the neckline and print quieter so the accessory remains a clear focal point.; [K080] Crossbody and sling bags suit movement-heavy outfits, so keep the top practical and easy to layer. A tee, tank, hoodie, sweater, or casual shirt works for travel and weekends; a clean button-down or fine knit can make a leather crossbody feel city-polished. If the strap crosses a busy print, choose a simpler neckline or smaller-scale pattern so the bag does not visually fight with the top.
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=N/A
- Selection / interpretation: deterministic median exact-support record

## Trace utilization high

- Case / generator / condition: `case-e092494ef218951a` / `ministral-3:14b-instruct-2512-q4_K_M` / `rule_rag`
- Target category; trace size: bags; 2
- Explanation: The **block box black leather shoulder bag** complements your **yoins black rivet embellished short boots** by balancing their bold embellishments with a structured, cohesive silhouette. Its sleek leather form aligns with the boots’ polished aesthetic while providing visual substance to avoid contrast overload, ensuring the bag feels intentional rather than disconnected. [K096] [K181]
- Extracted claim: The block box black leather shoulder bag complements your yoins black rivet embellished short boots.
- Evidence/reference: [K096] Boots support bags with a little visual substance. Shoulder bags, crossbodies, totes, structured handbags, and hobos all work; sleek boots pair more easily with structured bags, while biker or chunky boots can take softer or more casual bags. Repeating leather colour can create cohesion, but it is not required—especially when the outfit already contains several brown, black, or metallic elements.; [K181] Boots can support either practical or polished bags, but the bag should acknowledge the boot's visual weight. Chunky, combat, or biker boots work with crossbodies, shoulder bags, totes, and soft hobos for everyday wear; sleek ankle or riding boots can take a more structured shoulder bag or top-handle for city polish. Avoid pairing very rugged boots with a tiny ornate bag unless the contrast is intentional, because the accessory can look visually disconnected.
- Verifier outcome: exact=supported; full_kb=supported; common=N/A
- Selection / interpretation: deterministic 75%-quantile rule use

## Trace utilization low

- Case / generator / condition: `case-23bf6bb2a780b76b` / `gemma4:12b` / `rule_rag`
- Target category; trace size: tops; 3
- Explanation: The nly trend perfect neckline top is an excellent choice to pair with your shoes. Because Mary Janes provide a polished base, a clean neckline and controlled shape mirror the precision of the footwear while ensuring the look remains intentional. This specific top avoids excessive embellishment, allowing the refined silhouette of the pump to remain the focal point of the ensemble without the outfit appearing overly severe. [K131] [K038]
- Extracted claim: This specific top avoids excessive embellishment.
- Evidence/reference: [K131] Court shoes, slingbacks, and pointed pumps already sharpen an outfit, so the top can be simple and still look intentional. For work, choose a blouse, button-up, fine knit, or polished tank; for dinner, a silk camisole or fitted top works without needing heavy decoration. A clean neckline and controlled sleeve shape mirror the precision of the shoe, while softer fabric can prevent the outfit from becoming too severe.; [K038] With loafers, ballet flats, Mary Janes, or other refined flats, build upward with a clean top rather than athletic performance wear. A button-down, polo, fine knit, cardigan, or simple tee works for daytime; for office or dinner, use a blouse or sharper shirt and keep logos minimal. Refined flat shoes already provide a polished base, so the top can stay understated while still looking intentional.
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=N/A
- Selection / interpretation: deterministic 25%-quantile rule use

## Robustness: gemma4:12b

- Case / generator / condition: `case-31f8504807639231` / `gemma4:12b` / `no_rag`
- Target category; trace size: bags; 1
- Explanation: The bucket crossbody bag is the perfect choice to complete your look. This specific piece provides a functional contrast to the structured silhouette of the yoins lapel woolen duster coat. By pairing these items together, you create a balanced aesthetic that complements the coat's sophisticated design while offering a practical way to carry your essentials.
- Extracted claim: The bucket crossbody bag and the yoins lapel woolen duster coat create a balanced aesthetic.
- Evidence/reference: [K122] Coats and trenches usually look most coherent with bags that can physically sit over or under the outer layer without distortion. Structured shoulder bags, totes, top handles, and crossbodies work for commuting; choose strap lengths that clear bulky sleeves, and avoid tiny bags disappearing against very voluminous coats unless that contrast is deliberate. A clean leather or structured fabric bag helps maintain polish when the outerwear is heavy.
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=N/A
- Selection / interpretation: generator-median real record

## Robustness: llama3.1:8b-instruct-q8_0

- Case / generator / condition: `case-31f8504807639231` / `llama3.1:8b-instruct-q8_0` / `no_rag`
- Target category; trace size: bags; 1
- Explanation: The bucket crossbody bag complements the yoins lapel woolen duster coat by adding a relaxed, effortless touch to the overall look. Its compact size and casual design won't compete with the coat's statement lapels, allowing the two pieces to coexist harmoniously. This understated bag will also free up hands, making it perfect for everyday wear.
- Extracted claim: The bucket crossbody bag complements the yoins lapel woolen duster coat.
- Evidence/reference: [K122] Coats and trenches usually look most coherent with bags that can physically sit over or under the outer layer without distortion. Structured shoulder bags, totes, top handles, and crossbodies work for commuting; choose strap lengths that clear bulky sleeves, and avoid tiny bags disappearing against very voluminous coats unless that contrast is deliberate. A clean leather or structured fabric bag helps maintain polish when the outerwear is heavy.
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=N/A
- Selection / interpretation: generator-median real record

## Robustness: ministral-3:14b-instruct-2512-q4_K_M

- Case / generator / condition: `case-b84372a3927144c5` / `ministral-3:14b-instruct-2512-q4_K_M` / `no_rag`
- Target category; trace size: tops; 2
- Explanation: The **vero moda halter top** complements the **vince camuto tapered strap high heel sandal** by offering a sleek, structured silhouette that balances the sandal’s strappy design. Its minimalist cut avoids overwhelming the footwear’s refined details while maintaining a cohesive, polished look through shared elegance in line and proportion. The halter style subtly elongates the leg, harmonizing with the sandal’s elongated strap for a streamlined aesthetic.
- Extracted claim: The vero moda halter top balances the sandal's strappy design.
- Evidence/reference: [K041] With sandals, wedges, espadrilles, or clogs, keep the upper half seasonally light. Linen or cotton shirts, tanks, wrap tops, simple tees, and light blouses suit holiday and daytime wear; a silk camisole or refined sleeveless top works for a summer dinner. Natural textures and lighter colours usually make the outfit feel coherent, while a more structured top can stop platform or chunky sandals from reading too beachy.
- Verifier outcome: exact=not_supported; full_kb=not_supported; common=N/A
- Selection / interpretation: generator-median real record
