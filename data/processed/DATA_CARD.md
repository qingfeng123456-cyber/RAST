# Prepared dataset

Source: blanchon/opencs2_dataset_demo at revision 912a363098feec3f220e41d68634d2f71676c28b.

Map: de_dust2; demos: 53; match groups: 53; rounds: 1127; windows: 98203.

Targets: current anonymous regional counts of alive T actors not exposed by the specified radar proxy. Only training matches fit the spatial partition. Inputs do not contain hidden positions or enemy IDs.

Full sampled round prefixes include buy time. The 2 Hz grid is refined only with public deaths and friendly smoke lifecycle changes; hidden enemy smoke timing never creates observations. This is not every game tick or full audiovisual information.

All current professional event demos assigned band=4 by user convention, one level above the highest ordinary-player band=3. This constant-band subset cannot validate cross-rank generalization. Chronological Elo is a separate proxy, never an official rank.

Friendly smoke detonation/expiry history only; no hidden enemy smoke positions, volumetric vision reconstruction or movement blockade.

All logged player_sound step=true events assumed heard and localized to training-fitted regions. Idealized team audio: no walls, distance attenuation, masking, HRTF or hearing uncertainty. No enemy identifiers or exact sound coordinates reach the model.

Input histories with at least one sampled exposure: 51.40%. Metadata clock corrections: 53.

Splits and per-demo audits: split_matches.json, demo_audit.csv, summary.json. No expert tactical labels or human-visibility ground truth are provided. This subset is not the full upstream dataset and not exclusively top-tier play.


## Rebalanced whole-match split

User-requested group-disjoint rebalance: selected old test matches are now training data. Validation groups stay fixed. This is not a strictly chronological extrapolation split. All methods and seeds must share this version.

Moved 3 entire matches, 75 rounds and 6759 prefixes (20.0012% of old test prefixes) into training. Region centers were refitted on the new training groups, then every region-dependent input and label was rebuilt from the preserved parser-7 frame and audio caches. The old 483-round and 53-match versions remain unchanged. No data was downloaded or reparsed; no model was trained.


## Professional event band encoding

Ordinary-player bands use 0..3; professional event demos use band 4. Input bands are divided by 4, so ordinary highest band 3 is 0.75 and professional band 4 is 1.00. A separate mask marks missing bands. No actual Valve ranks were invented. The legacy professional-only cohort was encoded as 3/3=1, so this migration changes endpoint band metadata and the cache contract while keeping every frame exactly unchanged. All current samples have professional event band 4; this cohort cannot measure skill-band effects across ordinary ranks. See skill_band_migration.json.
