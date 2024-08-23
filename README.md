[bestsellers.with.categories.csv](https://github.com/user-attachments/files/16715345/bestsellers.with.categories.csv)# Amazon-Bestselling-Books-Analysis-with-Python
# Description-
This project aims to analyze Amazon bestseller data from 2009 to 2019 using Python. By examining factors such as genre, author, publication year, and user ratings, we will uncover trends, patterns, and insights into book sales and reader preferences. Employing data cleaning, statistical analysis, and visualization techniques, we seek to identify top-selling genres, authors, and understand the factors influencing book success. Ultimately, this analysis will provide valuable insights for authors, publishers, and readers alike.
# Methodology-
![image](https://github.com/user-attachments/assets/e9fcb606-c3d1-4101-9939-26e64873c43a)

# Sample Dataset-



[UploadinName,Author,User Rating,Reviews,Price,Year,Genre
10-Day Green Smoothie Cleanse,JJ Smith,4.7,17350,8,2016,Non Fiction
11/22/63: A Novel,Stephen King,4.6,2052,22,2011,Fiction
12 Rules for Life: An Antidote to Chaos,Jordan B. Peterson,4.7,18979,15,2018,Non Fiction
1984 (Signet Classics),George Orwell,4.7,21424,6,2017,Fiction
"5,000 Awesome Facts (About Everything!) (National Geographic Kids)",National Geographic Kids,4.8,7665,12,2019,Non Fiction
A Dance with Dragons (A Song of Ice and Fire),George R. R. Martin,4.4,12643,11,2011,Fiction
A Game of Thrones / A Clash of Kings / A Storm of Swords / A Feast of Crows / A Dance with Dragons,George R. R. Martin,4.7,19735,30,2014,Fiction
A Gentleman in Moscow: A Novel,Amor Towles,4.7,19699,15,2017,Fiction
"A Higher Loyalty: Truth, Lies, and Leadership",James Comey,4.7,5983,3,2018,Non Fiction
A Man Called Ove: A Novel,Fredrik Backman,4.6,23848,8,2016,Fiction
A Man Called Ove: A Novel,Fredrik Backman,4.6,23848,8,2017,Fiction
A Patriot's History of the United States: From Columbus's Great Discovery to the War on Terror,Larry Schweikart,4.6,460,2,2010,Non Fiction
A Stolen Life: A Memoir,Jaycee Dugard,4.6,4149,32,2011,Non Fiction
A Wrinkle in Time (Time Quintet),Madeleine L'Engle,4.5,5153,5,2018,Fiction
"Act Like a Lady, Think Like a Man: What Men Really Think About Love, Relationships, Intimacy, and Commitment",Steve Harvey,4.6,5013,17,2009,Non Fiction
"Adult Coloring Book Designs: Stress Relief Coloring Book: Garden Designs, Mandalas, Animals, and Paisley Patterns",Adult Coloring Book Designs,4.5,2313,4,2016,Non Fiction
Adult Coloring Book: Stress Relieving Animal Designs,Blue Star Coloring,4.6,2925,6,2015,Non Fiction
Adult Coloring Book: Stress Relieving Patterns,Blue Star Coloring,4.4,2951,6,2015,Non Fiction
"Adult Coloring Books: A Coloring Book for Adults Featuring Mandalas and Henna Inspired Flowers, Animals, and Paisley…",Coloring Books for Adults,4.5,2426,8,2015,Non Fiction
Alexander Hamilton,Ron Chernow,4.8,9198,13,2016,Non Fiction
All the Light We Cannot See,Anthony Doerr,4.6,36348,14,2014,Fiction
All the Light We Cannot See,Anthony Doerr,4.6,36348,14,2015,Fiction
Allegiant,Veronica Roth,3.9,6310,13,2013,Fiction
American Sniper: The Autobiography of the Most Lethal Sniper in U.S. Military History,Chris Kyle,4.6,15921,9,2015,Non Fiction
And the Mountains Echoed,Khaled Hosseini,4.3,12159,13,2013,Fiction
Arguing with Idiots: How to Stop Small Minds and Big Government,Glenn Beck,4.6,798,5,2009,Non Fiction
Astrophysics for People in a Hurry,Neil deGrasse Tyson,4.7,9374,9,2017,Non Fiction
"Autobiography of Mark Twain, Vol. 1",Mark Twain,4.2,491,14,2010,Non Fiction
Baby Touch and Feel: Animals,DK,4.6,5360,5,2015,Non Fiction
Balance (Angie's Extreme Stress Menders),Angie Grace,4.6,1909,11,2015,Non Fiction
Barefoot Contessa Foolproof: Recipes You Can Trust: A Cookbook,Ina Garten,4.8,1296,24,2012,Non Fiction
"Barefoot Contessa, How Easy Is That?: Fabulous Recipes & Easy Tips",Ina Garten,4.7,615,21,2010,Non Fiction
Becoming,Michelle Obama,4.8,61133,11,2018,Non Fiction
Becoming,Michelle Obama,4.8,61133,11,2019,Non Fiction
Being Mortal: Medicine and What Matters in the End,Atul Gawande,4.8,11113,15,2015,Non Fiction
Between the World and Me,Ta-Nehisi Coates,4.7,10070,13,2015,Non Fiction
Between the World and Me,Ta-Nehisi Coates,4.7,10070,13,2016,Non Fiction
Born to Run,Bruce Springsteen,4.7,3729,18,2016,Non Fiction
"Breaking Dawn (The Twilight Saga, Book 4)",Stephenie Meyer,4.6,9769,13,2009,Fiction
"Broke: The Plan to Restore Our Trust, Truth and Treasure",Glenn Beck,4.5,471,8,2010,Non Fiction
"Brown Bear, Brown Bear, What Do You See?",Bill Martin Jr.,4.9,14344,5,2017,Fiction
"Brown Bear, Brown Bear, What Do You See?",Bill Martin Jr.,4.9,14344,5,2019,Fiction
"Cabin Fever (Diary of a Wimpy Kid, Book 6)",Jeff Kinney,4.8,4505,0,2011,Fiction
Calm the F*ck Down: An Irreverent Adult Coloring Book (Irreverent Book Series),Sasha O'Hara,4.6,10369,4,2016,Non Fiction
Can't Hurt Me: Master Your Mind and Defy the Odds,David Goggins,4.8,16244,18,2019,Non Fiction
Capital in the Twenty First Century,Thomas Piketty,4.5,2884,28,2014,Non Fiction
Catching Fire (The Hunger Games),Suzanne Collins,4.7,22614,11,2010,Fiction
Catching Fire (The Hunger Games),Suzanne Collins,4.7,22614,11,2011,Fiction
Catching Fire (The Hunger Games),Suzanne Collins,4.7,22614,11,2012,Fiction
Cravings: Recipes for All the Food You Want to Eat: A Cookbook,Chrissy Teigen,4.7,4761,16,2016,Non Fiction
Crazy Love: Overwhelmed by a Relentless God,Francis Chan,4.7,1542,14,2009,Non Fiction
Crazy Love: Overwhelmed by a Relentless God,Francis Chan,4.7,1542,14,2010,Non Fiction
Crazy Love: Overwhelmed by a Relentless God,Francis Chan,4.7,1542,14,2011,Non Fiction
Crazy Rich Asians (Crazy Rich Asians Trilogy),Kevin Kwan,4.3,6143,8,2018,Fiction
Creative Haven Creative Cats Coloring Book (Adult Coloring),Marjorie Sarnat,4.8,4022,4,2015,Non Fiction
Creative Haven Owls Coloring Book (Adult Coloring),Marjorie Sarnat,4.8,3871,5,2015,Non Fiction
Cutting for Stone,Abraham Verghese,4.6,4866,11,2010,Fiction
Cutting for Stone,Abraham Verghese,4.6,4866,11,2011,Fiction
"Daring Greatly: How the Courage to Be Vulnerable Transforms the Way We Live, Love, Parent, and Lead",Brené Brown,4.8,1329,10,2013,Non Fiction
"David and Goliath: Underdogs, Misfits, and the Art of Battling Giants",Malcolm Gladwell,4.4,4642,13,2013,Non Fiction
Dead And Gone: A Sookie Stackhouse Novel (Sookie Stackhouse/True Blood),Charlaine Harris,4.6,1541,4,2009,Fiction
"Dead in the Family (Sookie Stackhouse/True Blood, Book 10)",Charlaine Harris,4.3,1924,8,2010,Fiction
"Dead Reckoning (Sookie Stackhouse/True Blood, Book 11)",Charlaine Harris,4.2,2094,4,2011,Fiction
Dear Zoo: A Lift-the-Flap Book,Rod Campbell,4.8,10922,5,2015,Fiction
Dear Zoo: A Lift-the-Flap Book,Rod Campbell,4.8,10922,5,2016,Fiction
Dear Zoo: A Lift-the-Flap Book,Rod Campbell,4.8,10922,5,2017,Fiction
Dear Zoo: A Lift-the-Flap Book,Rod Campbell,4.8,10922,5,2018,Fiction
Decision Points,George W. Bush,4.6,2137,17,2010,Non Fiction
"Delivering Happiness: A Path to Profits, Passion, and Purpose",Tony Hsieh,4.6,1651,15,2010,Non Fiction
"Diagnostic and Statistical Manual of Mental Disorders, 5th Edition: DSM-5",American Psychiatric Association,4.5,6679,105,2013,Non Fiction
"Diagnostic and Statistical Manual of Mental Disorders, 5th Edition: DSM-5",American Psychiatric Association,4.5,6679,105,2014,Non Fiction
"Diary of a Wimpy Kid: Hard Luck, Book 8",Jeff Kinney,4.8,6812,0,2013,Fiction
Diary of a Wimpy Kid: The Last Straw (Book 3),Jeff Kinney,4.8,3837,15,2009,Fiction
Diary of a Wimpy Kid: The Long Haul,Jeff Kinney,4.8,6540,22,2014,Fiction
Difficult Riddles For Smart Kids: 300 Difficult Riddles And Brain Teasers Families Will Love (Books for Smart Kids),M Prefontaine,4.6,7955,5,2019,Non Fiction
Divergent,Veronica Roth,4.6,27098,15,2013,Fiction
Divergent,Veronica Roth,4.6,27098,15,2014,Fiction
Divergent / Insurgent,Veronica Roth,4.5,17684,6,2014,Fiction
"Divine Soul Mind Body Healing and Transmission System: The Divine Way to Heal You, Humanity, Mother Earth, and All…",Zhi Gang Sha,4.6,37,6,2009,Non Fiction
Doctor Sleep: A Novel,Stephen King,4.7,15845,13,2013,Fiction
"Dog Days (Diary of a Wimpy Kid, Book 4) (Volume 4)",Jeff Kinney,4.8,3181,12,2009,Fiction
Dog Man and Cat Kid: From the Creator of Captain Underpants (Dog Man #4),Dav Pilkey,4.9,5062,6,2018,Fiction
Dog Man: A Tale of Two Kitties: From the Creator of Captain Underpants (Dog Man #3),Dav Pilkey,4.9,4786,8,2017,Fiction
Dog Man: Brawl of the Wild: From the Creator of Captain Underpants (Dog Man #6),Dav Pilkey,4.9,7235,4,2018,Fiction
Dog Man: Brawl of the Wild: From the Creator of Captain Underpants (Dog Man #6),Dav Pilkey,4.9,7235,4,2019,Fiction
Dog Man: Fetch-22: From the Creator of Captain Underpants (Dog Man #8),Dav Pilkey,4.9,12619,8,2019,Fiction
Dog Man: For Whom the Ball Rolls: From the Creator of Captain Underpants (Dog Man #7),Dav Pilkey,4.9,9089,8,2019,Fiction
Dog Man: Lord of the Fleas: From the Creator of Captain Underpants (Dog Man #5),Dav Pilkey,4.9,5470,6,2018,Fiction
Double Down (Diary of a Wimpy Kid #11),Jeff Kinney,4.8,5118,20,2016,Fiction
Dover Creative Haven Art Nouveau Animal Designs Coloring Book (Creative Haven Coloring Books),Marty Noble,4.6,2134,5,2015,Non Fiction
Drive: The Surprising Truth About What Motivates Us,Daniel H. Pink,4.5,2525,16,2010,Non Fiction
Eat This Not That! Supermarket Survival Guide: The No-Diet Weight Loss Solution,David Zinczenko,4.5,720,1,2009,Non Fiction
"Eat This, Not That! Thousands of Simple Food Swaps that Can Save You 10, 20, 30 Pounds--or More!",David Zinczenko,4.3,956,14,2009,Non Fiction
"Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss, Revised Edition",Joel Fuhrman MD,4.5,6346,9,2011,Non Fiction
"Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss, Revised Edition",Joel Fuhrman MD,4.5,6346,9,2012,Non Fiction
Eclipse (Twilight Sagas),Stephenie Meyer,4.7,5505,7,2009,Fiction
Eclipse (Twilight),Stephenie Meyer,4.7,5505,18,2009,Fiction
Educated: A Memoir,Tara Westover,4.7,28729,15,2018,Non Fiction
Educated: A Memoir,Tara Westover,4.7,28729,15,2019,Non Fiction
"Enchanted Forest: An Inky Quest and Coloring book (Activity Books, Mindfulness and Meditation, Illustrated Floral Prints…",Johanna Basford,4.7,5413,9,2015,Non Fiction
Fahrenheit 451,Ray Bradbury,4.6,10721,8,2016,Fiction
Fahrenheit 451,Ray Bradbury,4.6,10721,8,2018,Fiction
Fantastic Beasts and Where to Find Them: The Original Screenplay (Harry Potter),J.K. Rowling,4.7,4370,15,2016,Fiction
Fear: Trump in the White House,Bob Woodward,4.4,6042,2,2018,Non Fiction
Fifty Shades Darker,E L James,4.4,23631,7,2012,Fiction
Fifty Shades Freed: Book Three of the Fifty Shades Trilogy (Fifty Shades of Grey Series) (English Edition),E L James,4.5,20262,11,2012,Fiction
Fifty Shades of Grey: Book One of the Fifty Shades Trilogy (Fifty Shades of Grey Series),E L James,3.8,47265,14,2012,Fiction
Fifty Shades of Grey: Book One of the Fifty Shades Trilogy (Fifty Shades of Grey Series),E L James,3.8,47265,14,2013,Fiction
Fifty Shades Trilogy (Fifty Shades of Grey / Fifty Shades Darker / Fifty Shades Freed),E L James,4.5,13964,32,2012,Fiction
Fire and Fury: Inside the Trump White House,Michael Wolff,4.2,13677,6,2018,Non Fiction
First 100 Words,Roger Priddy,4.7,17323,4,2014,Non Fiction
First 100 Words,Roger Priddy,4.7,17323,4,2015,Non Fiction
First 100 Words,Roger Priddy,4.7,17323,4,2016,Non Fiction
First 100 Words,Roger Priddy,4.7,17323,4,2017,Non Fiction
First 100 Words,Roger Priddy,4.7,17323,4,2018,Non Fiction
Food Rules: An Eater's Manual,Michael Pollan,4.4,1555,9,2010,Non Fiction
Frozen (Little Golden Book),RH Disney,4.7,3642,0,2014,Fiction
"Game Change: Obama and the Clintons, McCain and Palin, and the Race of a Lifetime",John Heilemann,4.4,1215,9,2010,Non Fiction
Game of Thrones Boxed Set: A Game of Thrones/A Clash of Kings/A Storm of Swords/A Feast for Crows,George R.R. Martin,4.6,5594,5,2011,Fiction
Game of Thrones Boxed Set: A Game of Thrones/A Clash of Kings/A Storm of Swords/A Feast for Crows,George R.R. Martin,4.6,5594,5,2012,Fiction
Game of Thrones Boxed Set: A Game of Thrones/A Clash of Kings/A Storm of Swords/A Feast for Crows,George R.R. Martin,4.6,5594,5,2013,Fiction
George Washington's Sacred Fire,Peter A. Lillback,4.5,408,20,2010,Non Fiction
George Washington's Secret Six: The Spy Ring That Saved the American Revolution,Brian Kilmeade,4.6,4799,16,2013,Non Fiction
Giraffes Can't Dance,Giles Andreae,4.8,14038,4,2015,Fiction
Giraffes Can't Dance,Giles Andreae,4.8,14038,4,2016,Fiction
Giraffes Can't Dance,Giles Andreae,4.8,14038,4,2017,Fiction
Giraffes Can't Dance,Giles Andreae,4.8,14038,4,2018,Fiction
Giraffes Can't Dance,Giles Andreae,4.8,14038,4,2019,Fiction
"Girl, Stop Apologizing: A Shame-Free Plan for Embracing and Achieving Your Goals",Rachel Hollis,4.6,7660,12,2019,Non Fiction
"Girl, Wash Your Face: Stop Believing the Lies About Who You Are So You Can Become Who You Were Meant to Be",Rachel Hollis,4.6,22288,12,2018,Non Fiction
"Girl, Wash Your Face: Stop Believing the Lies About Who You Are So You Can Become Who You Were Meant to Be",Rachel Hollis,4.6,22288,12,2019,Non Fiction
"Glenn Beck's Common Sense: The Case Against an Out-of-Control Government, Inspired by Thomas Paine",Glenn Beck,4.6,1365,11,2009,Non Fiction
Go Set a Watchman: A Novel,Harper Lee,3.6,14982,19,2015,Fiction
Go the F**k to Sleep,Adam Mansbach,4.8,9568,9,2011,Fiction
Going Rogue: An American Life,Sarah Palin,4.6,1636,6,2009,Non Fiction
Gone Girl,Gillian Flynn,4,57271,10,2012,Fiction
Gone Girl,Gillian Flynn,4,57271,10,2013,Fiction
Gone Girl,Gillian Flynn,4,57271,9,2014,Fiction
Good Days Start With Gratitude: A 52 Week Guide To Cultivate An Attitude Of Gratitude: Gratitude Journal,Pretty Simple Press,4.6,10141,6,2019,Non Fiction
Good to Great: Why Some Companies Make the Leap and Others Don't,Jim Collins,4.5,3457,14,2009,Non Fiction
Good to Great: Why Some Companies Make the Leap and Others Don't,Jim Collins,4.5,3457,14,2010,Non Fiction
Good to Great: Why Some Companies Make the Leap and Others Don't,Jim Collins,4.5,3457,14,2011,Non Fiction
Good to Great: Why Some Companies Make the Leap and Others Don't,Jim Collins,4.5,3457,14,2012,Non Fiction
Goodnight Moon,Margaret Wise Brown,4.8,8837,5,2017,Fiction
Goodnight Moon,Margaret Wise Brown,4.8,8837,5,2018,Fiction
Goodnight Moon,Margaret Wise Brown,4.8,8837,5,2019,Fiction
"Goodnight, Goodnight Construction Site (Hardcover Books for Toddlers, Preschool Books for Kids)",Sherri Duskey Rinker,4.9,7038,7,2012,Fiction
"Goodnight, Goodnight Construction Site (Hardcover Books for Toddlers, Preschool Books for Kids)",Sherri Duskey Rinker,4.9,7038,7,2013,Fiction
"Grain Brain: The Surprising Truth about Wheat, Carbs, and Sugar--Your Brain's Silent Killers",David Perlmutter MD,4.6,5972,10,2014,Non Fiction
Grey: Fifty Shades of Grey as Told by Christian (Fifty Shades of Grey Series),E L James,4.4,25624,14,2015,Fiction
Guts,Raina Telgemeier,4.8,5476,7,2019,Non Fiction
Hamilton: The Revolution,Lin-Manuel Miranda,4.9,5867,54,2016,Non Fiction
"Happy, Happy, Happy: My Life and Legacy as the Duck Commander",Phil Robertson,4.8,4148,11,2013,Non Fiction
"Harry Potter and the Chamber of Secrets: The Illustrated Edition (Harry Potter, Book 2)",J.K. Rowling,4.9,19622,30,2016,Fiction
"Harry Potter and the Cursed Child, Parts 1 & 2, Special Rehearsal Edition Script",J.K. Rowling,4,23973,12,2016,Fiction
"Harry Potter and the Goblet of Fire: The Illustrated Edition (Harry Potter, Book 4) (4)",J. K. Rowling,4.9,7758,18,2019,Fiction
"Harry Potter and the Prisoner of Azkaban: The Illustrated Edition (Harry Potter, Book 3)",J.K. Rowling,4.9,3146,30,2017,Fiction
"Harry Potter and the Sorcerer's Stone: The Illustrated Edition (Harry Potter, Book 1)",J.K. Rowling,4.9,10052,22,2016,Fiction
Harry Potter Coloring Book,Scholastic,4.7,3564,9,2015,Non Fiction
Harry Potter Paperback Box Set (Books 1-7),J. K. Rowling,4.8,13471,52,2016,Fiction
Have a Little Faith: A True Story,Mitch Albom,4.8,1930,4,2009,Non Fiction
Heaven is for Real: A Little Boy's Astounding Story of His Trip to Heaven and Back,Todd Burpo,4.7,15779,10,2011,Non Fiction
Heaven is for Real: A Little Boy's Astounding Story of His Trip to Heaven and Back,Todd Burpo,4.7,15779,10,2012,Non Fiction
Hillbilly Elegy: A Memoir of a Family and Culture in Crisis,J. D. Vance,4.4,15526,14,2016,Non Fiction
Hillbilly Elegy: A Memoir of a Family and Culture in Crisis,J. D. Vance,4.4,15526,14,2017,Non Fiction
Homebody: A Guide to Creating Spaces You Never Want to Leave,Joanna Gaines,4.8,3776,22,2018,Non Fiction
How to Win Friends & Influence People,Dale Carnegie,4.7,25001,11,2014,Non Fiction
How to Win Friends & Influence People,Dale Carnegie,4.7,25001,11,2015,Non Fiction
How to Win Friends & Influence People,Dale Carnegie,4.7,25001,11,2016,Non Fiction
How to Win Friends & Influence People,Dale Carnegie,4.7,25001,11,2017,Non Fiction
How to Win Friends & Influence People,Dale Carnegie,4.7,25001,11,2018,Non Fiction
Howard Stern Comes Again,Howard Stern,4.3,5272,16,2019,Non Fiction
Humans of New York,Brandon Stanton,4.8,3490,15,2013,Non Fiction
Humans of New York,Brandon Stanton,4.8,3490,15,2014,Non Fiction
Humans of New York : Stories,Brandon Stanton,4.9,2812,17,2015,Non Fiction
"Hyperbole and a Half: Unfortunate Situations, Flawed Coping Mechanisms, Mayhem, and Other Things That Happened",Allie Brosh,4.7,4896,17,2013,Non Fiction
"I Am Confident, Brave & Beautiful: A Coloring Book for Girls",Hopscotch Girls,4.8,9737,7,2019,Non Fiction
"I, Alex Cross",James Patterson,4.6,1320,7,2009,Fiction
If Animals Kissed Good Night,Ann Whitford Paul,4.8,16643,4,2017,Fiction
If Animals Kissed Good Night,Ann Whitford Paul,4.8,16643,4,2019,Fiction
If I Stay,Gayle Forman,4.3,7153,9,2014,Fiction
"In the Garden of Beasts: Love, Terror, and an American Family in Hitler's Berlin",Eric Larson,4.4,4571,21,2011,Non Fiction
Inferno,Dan Brown,4.1,29651,14,2013,Fiction
Inheritance: Book IV (Inheritance Cycle),Christopher Paolini,4.6,5299,20,2011,Fiction
Instant Pot Pressure Cooker Cookbook: 500 Everyday Recipes for Beginners and Advanced Users. Try Easy and Healthy…,Jennifer Smith,4.4,7396,13,2019,Non Fiction
Instant Pot Pressure Cooker Cookbook: 500 Everyday Recipes for Beginners and Advanced Users. Try Easy and Healthy…,Jennifer Smith,4.4,7396,13,2018,Non Fiction
It's Not Supposed to Be This Way: Finding Unexpected Strength When Disappointments Leave You Shattered,Lysa TerKeurst,4.8,7062,12,2019,Non Fiction
Jesus Calling: Enjoying Peace in His Presence (with Scripture References),Sarah Young,4.9,19576,8,2011,Non Fiction
Jesus Calling: Enjoying Peace in His Presence (with Scripture References),Sarah Young,4.9,19576,8,2012,Non Fiction
Jesus Calling: Enjoying Peace in His Presence (with Scripture References),Sarah Young,4.9,19576,8,2013,Non Fiction
Jesus Calling: Enjoying Peace in His Presence (with Scripture References),Sarah Young,4.9,19576,8,2014,Non Fiction
Jesus Calling: Enjoying Peace in His Presence (with Scripture References),Sarah Young,4.9,19576,8,2015,Non Fiction
Jesus Calling: Enjoying Peace in His Presence (with Scripture References),Sarah Young,4.9,19576,8,2016,Non Fiction
JOURNEY TO THE ICE P,RH Disney,4.6,978,0,2014,Fiction
Joyland (Hard Case Crime),Stephen King,4.5,4748,12,2013,Fiction
Killers of the Flower Moon: The Osage Murders and the Birth of the FBI,David Grann,4.6,8393,17,2017,Non Fiction
Killing Jesus (Bill O'Reilly's Killing Series),Bill O'Reilly,4.5,11391,12,2013,Non Fiction
Killing Kennedy: The End of Camelot,Bill O'Reilly,4.6,8634,25,2012,Non Fiction
Killing Lincoln: The Shocking Assassination that Changed America Forever (Bill O'Reilly's Killing Series),Bill O'Reilly,4.7,9342,10,2011,Non Fiction
Killing Lincoln: The Shocking Assassination that Changed America Forever (Bill O'Reilly's Killing Series),Bill O'Reilly,4.7,9342,10,2012,Non Fiction
Killing Patton: The Strange Death of World War II's Most Audacious General (Bill O'Reilly's Killing Series),Bill O'Reilly,4.6,10927,6,2014,Non Fiction
Killing Reagan: The Violent Assault That Changed a Presidency (Bill O'Reilly's Killing Series),Bill O'Reilly,4.6,5235,5,2015,Non Fiction
Killing the Rising Sun: How America Vanquished World War II Japan (Bill O'Reilly's Killing Series),Bill O'Reilly,4.8,8916,6,2016,Non Fiction
Kitchen Confidential Updated Edition: Adventures in the Culinary Underbelly (P.S.),Anthony Bourdain,4.8,2507,8,2018,Non Fiction
Knock-Knock Jokes for Kids,Rob Elliott,4.5,3673,4,2013,Non Fiction
Knock-Knock Jokes for Kids,Rob Elliott,4.5,3673,4,2014,Non Fiction
Knock-Knock Jokes for Kids,Rob Elliott,4.5,3673,4,2015,Non Fiction
"Last Week Tonight with John Oliver Presents A Day in the Life of Marlon Bundo (Better Bundo Book, LGBT Childrens Book)",Jill Twiss,4.9,11881,13,2018,Fiction
Laugh-Out-Loud Jokes for Kids,Rob Elliott,4.6,6990,4,2013,Non Fiction
Laugh-Out-Loud Jokes for Kids,Rob Elliott,4.6,6990,4,2014,Non Fiction
Laugh-Out-Loud Jokes for Kids,Rob Elliott,4.6,6990,4,2015,Non Fiction
Laugh-Out-Loud Jokes for Kids,Rob Elliott,4.6,6990,4,2016,Non Fiction
Laugh-Out-Loud Jokes for Kids,Rob Elliott,4.6,6990,4,2017,Non Fiction
"Lean In: Women, Work, and the Will to Lead",Sheryl Sandberg,4.5,6132,13,2013,Non Fiction
Leonardo da Vinci,Walter Isaacson,4.5,3014,21,2017,Non Fiction
Lettering and Modern Calligraphy: A Beginner's Guide: Learn Hand Lettering and Brush Lettering,Paper Peony Press,4.4,7550,6,2018,Non Fiction
Liberty and Tyranny: A Conservative Manifesto,Mark R. Levin,4.8,3828,15,2009,Non Fiction
Life,Keith Richards,4.5,2752,18,2010,Non Fiction
Little Bee: A Novel,Chris Cleave,4.1,1467,10,2010,Fiction
Little Blue Truck,Alice Schertle,4.9,1884,0,2014,Fiction
Little Fires Everywhere,Celeste Ng,4.5,25706,12,2018,Fiction
Looking for Alaska,John Green,4.5,8491,7,2014,Fiction
"Love Wins: A Book About Heaven, Hell, and the Fate of Every Person Who Ever Lived",Rob Bell,4.2,1649,13,2011,Non Fiction
Love You Forever,Robert Munsch,4.8,18613,5,2014,Fiction
Love You Forever,Robert Munsch,4.8,18613,5,2015,Fiction
Magnolia Table: A Collection of Recipes for Gathering,Joanna Gaines,4.8,9867,16,2018,Non Fiction
Make It Ahead: A Barefoot Contessa Cookbook,Ina Garten,4.5,1386,20,2014,Non Fiction
Make Your Bed: Little Things That Can Change Your Life...And Maybe the World,Admiral William H. McRaven,4.7,10199,11,2017,Non Fiction
"Mastering the Art of French Cooking, Vol. 2",Julia Child,4.8,2926,27,2009,Non Fiction
Milk and Honey,Rupi Kaur,4.7,17739,8,2016,Non Fiction
Milk and Honey,Rupi Kaur,4.7,17739,8,2017,Non Fiction
Milk and Honey,Rupi Kaur,4.7,17739,8,2018,Non Fiction
Milk and Vine: Inspirational Quotes From Classic Vines,Adam Gasiewski,4.4,3113,6,2017,Non Fiction
Mindset: The New Psychology of Success,Carol S. Dweck,4.6,5542,10,2014,Non Fiction
Mindset: The New Psychology of Success,Carol S. Dweck,4.6,5542,10,2015,Non Fiction
Mindset: The New Psychology of Success,Carol S. Dweck,4.6,5542,10,2016,Non Fiction
Mockingjay (The Hunger Games),Suzanne Collins,4.5,26741,8,2010,Fiction
Mockingjay (The Hunger Games),Suzanne Collins,4.5,26741,8,2011,Fiction
Mockingjay (The Hunger Games),Suzanne Collins,4.5,26741,8,2012,Fiction
"National Geographic Kids Why?: Over 1,111 Answers to Everything",Crispin Boyer,4.8,5347,16,2019,Non Fiction
National Geographic Little Kids First Big Book of Why (National Geographic Little Kids First Big Books),Amy Shields,4.8,7866,11,2019,Non Fiction
New Moon (The Twilight Saga),Stephenie Meyer,4.6,5680,10,2009,Fiction
Night (Night),Elie Wiesel,4.7,5178,9,2016,Non Fiction
No Easy Day: The Autobiography of a Navy Seal: The Firsthand Account of the Mission That Killed Osama Bin Laden,Mark Owen,4.6,8093,14,2012,Non Fiction
Obama: An Intimate Portrait,Pete Souza,4.9,3192,22,2017,Non Fiction
"Oh, the Places You'll Go!",Dr. Seuss,4.9,21834,8,2012,Fiction
"Oh, the Places You'll Go!",Dr. Seuss,4.9,21834,8,2013,Fiction
"Oh, the Places You'll Go!",Dr. Seuss,4.9,21834,8,2014,Fiction
"Oh, the Places You'll Go!",Dr. Seuss,4.9,21834,8,2015,Fiction
"Oh, the Places You'll Go!",Dr. Seuss,4.9,21834,8,2016,Fiction
"Oh, the Places You'll Go!",Dr. Seuss,4.9,21834,8,2017,Fiction
"Oh, the Places You'll Go!",Dr. Seuss,4.9,21834,8,2018,Fiction
"Oh, the Places You'll Go!",Dr. Seuss,4.9,21834,8,2019,Fiction
Old School (Diary of a Wimpy Kid #10),Jeff Kinney,4.8,6169,7,2015,Fiction
Olive Kitteridge,Elizabeth Strout,4.2,4519,12,2009,Fiction
One Thousand Gifts: A Dare to Live Fully Right Where You Are,Ann Voskamp,4.6,3163,13,2011,Non Fiction
One Thousand Gifts: A Dare to Live Fully Right Where You Are,Ann Voskamp,4.6,3163,13,2012,Non Fiction
"Option B: Facing Adversity, Building Resilience, and Finding Joy",Sheryl Sandberg,4.5,1831,9,2017,Non Fiction
Origin: A Novel (Robert Langdon),Dan Brown,4.3,18904,13,2017,Fiction
Orphan Train,Christina Baker Kline,4.6,21930,11,2014,Fiction
Outliers: The Story of Success,Malcolm Gladwell,4.6,10426,20,2009,Non Fiction
Outliers: The Story of Success,Malcolm Gladwell,4.6,10426,20,2010,Non Fiction
P is for Potty! (Sesame Street) (Lift-the-Flap),Naomi Kleinberg,4.7,10820,5,2018,Non Fiction
P is for Potty! (Sesame Street) (Lift-the-Flap),Naomi Kleinberg,4.7,10820,5,2019,Non Fiction
Percy Jackson and the Olympians Paperback Boxed Set (Books 1-3),Rick Riordan,4.8,548,2,2010,Fiction
Player's Handbook (Dungeons & Dragons),Wizards RPG Team,4.8,16990,27,2017,Fiction
Player's Handbook (Dungeons & Dragons),Wizards RPG Team,4.8,16990,27,2018,Fiction
Player's Handbook (Dungeons & Dragons),Wizards RPG Team,4.8,16990,27,2019,Fiction
Pokémon Deluxe Essential Handbook: The Need-to-Know Stats and Facts on Over 700 Pokémon,Scholastic,4.7,3503,9,2016,Fiction
Proof of Heaven: A Neurosurgeon's Journey into the Afterlife,Eben Alexander,4.3,13616,10,2012,Non Fiction
Proof of Heaven: A Neurosurgeon's Journey into the Afterlife,Eben Alexander,4.3,13616,10,2013,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2009,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2010,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2011,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2012,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2013,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2014,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2015,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2016,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2017,Non Fiction
"Publication Manual of the American Psychological Association, 6th Edition",American Psychological Association,4.5,8580,46,2018,Non Fiction
Puppy Birthday to You! (Paw Patrol) (Little Golden Book),Golden Books,4.8,4757,4,2017,Fiction
Quiet: The Power of Introverts in a World That Can't Stop Talking,Susan Cain,4.6,10009,20,2012,Non Fiction
Quiet: The Power of Introverts in a World That Can't Stop Talking,Susan Cain,4.6,10009,7,2013,Non Fiction
Radical: Taking Back Your Faith from the American Dream,David Platt,4.7,1985,9,2010,Non Fiction
Radical: Taking Back Your Faith from the American Dream,David Platt,4.7,1985,9,2011,Non Fiction
Ready Player One: A Novel,Ernest Cline,4.6,22536,12,2017,Fiction
Ready Player One: A Novel,Ernest Cline,4.6,22536,12,2018,Fiction
Rush Revere and the Brave Pilgrims: Time-Travel Adventures with Exceptional Americans (1),Rush Limbaugh,4.9,7150,12,2013,Fiction
Rush Revere and the First Patriots: Time-Travel Adventures With Exceptional Americans (2),Rush Limbaugh,4.9,3836,12,2014,Fiction
"Salt, Fat, Acid, Heat: Mastering the Elements of Good Cooking",Samin Nosrat,4.8,7802,20,2018,Non Fiction
"Salt, Fat, Acid, Heat: Mastering the Elements of Good Cooking",Samin Nosrat,4.8,7802,20,2019,Non Fiction
Sarah's Key,Tatiana de Rosnay,4.6,3619,10,2010,Fiction
"School Zone - Big Preschool Workbook - Ages 4 and Up, Colors, Shapes, Numbers 1-10, Alphabet, Pre-Writing, Pre-Reading…",School Zone,4.8,23047,6,2018,Non Fiction
"School Zone - Big Preschool Workbook - Ages 4 and Up, Colors, Shapes, Numbers 1-10, Alphabet, Pre-Writing, Pre-Reading…",School Zone,4.8,23047,6,2019,Non Fiction
"Secret Garden: An Inky Treasure Hunt and Coloring Book (For Adults, mindfulness coloring)",Johanna Basford,4.7,9366,9,2015,Non Fiction
Sh*t My Dad Says,Justin Halpern,4.7,1265,11,2010,Non Fiction
Ship of Fools: How a Selfish Ruling Class Is Bringing America to the Brink of Revolution,Tucker Carlson,4.8,3923,16,2018,Non Fiction
Shred: The Revolutionary Diet: 6 Weeks 4 Inches 2 Sizes,Ian K. Smith M.D.,4.1,2272,6,2013,Non Fiction
Sookie Stackhouse,Charlaine Harris,4.7,973,25,2009,Fiction
"Soul Healing Miracles: Ancient and New Sacred Wisdom, Knowledge, and Practical Techniques for Healing the Spiritual…",Zhi Gang Sha,4.6,220,17,2013,Non Fiction
Steve Jobs,Walter Isaacson,4.6,7827,20,2011,Non Fiction
Steve Jobs,Walter Isaacson,4.6,7827,20,2012,Non Fiction
Strange Planet (Strange Planet Series),Nathan W. Pyle,4.9,9382,6,2019,Fiction
StrengthsFinder 2.0,Gallup,4,5069,17,2009,Non Fiction
StrengthsFinder 2.0,Gallup,4,5069,17,2010,Non Fiction
StrengthsFinder 2.0,Gallup,4,5069,17,2011,Non Fiction
StrengthsFinder 2.0,Gallup,4,5069,17,2012,Non Fiction
StrengthsFinder 2.0,Gallup,4,5069,17,2013,Non Fiction
StrengthsFinder 2.0,Gallup,4,5069,17,2014,Non Fiction
StrengthsFinder 2.0,Gallup,4,5069,17,2015,Non Fiction
StrengthsFinder 2.0,Gallup,4,5069,17,2016,Non Fiction
StrengthsFinder 2.0,Gallup,4,5069,17,2017,Non Fiction
"Super Freakonomics: Global Cooling, Patriotic Prostitutes, and Why Suicide Bombers Should Buy Life Insurance",Steven D. Levitt,4.5,1583,18,2009,Non Fiction
Switch: How to Change Things When Change Is Hard,Chip Heath,4.6,1907,13,2010,Non Fiction
Sycamore Row (Jake Brigance),John Grisham,4.5,23114,18,2013,Fiction
Teach Like a Champion: 49 Techniques that Put Students on the Path to College,Doug Lemov,4.4,637,20,2010,Non Fiction
Teach Like a Champion: 49 Techniques that Put Students on the Path to College,Doug Lemov,4.4,637,20,2011,Non Fiction
The 17 Day Diet: A Doctor's Plan Designed for Rapid Results,Mike Moreno,4.3,2314,22,2011,Non Fiction
"The 4 Hour Body: An Uncommon Guide to Rapid Fat Loss, Incredible Sex and Becoming Superhuman",Timothy Ferriss,4.3,4587,21,2011,Non Fiction
The 5 Love Languages: The Secret to Love That Lasts,Gary Chapman,4.7,3477,28,2010,Non Fiction
The 5 Love Languages: The Secret to Love That Lasts,Gary Chapman,4.7,3477,28,2011,Non Fiction
The 5 Love Languages: The Secret to Love That Lasts,Gary Chapman,4.7,3477,28,2012,Non Fiction
The 5 Love Languages: The Secret to Love That Lasts,Gary Chapman,4.7,3477,28,2013,Non Fiction
The 5 Love Languages: The Secret to Love That Lasts,Gary Chapman,4.7,3477,28,2014,Non Fiction
The 5 Love Languages: The Secret to Love that Lasts,Gary Chapman,4.8,25554,8,2015,Non Fiction
The 5 Love Languages: The Secret to Love that Lasts,Gary Chapman,4.8,25554,8,2016,Non Fiction
The 5 Love Languages: The Secret to Love that Lasts,Gary Chapman,4.8,25554,8,2017,Non Fiction
The 5 Love Languages: The Secret to Love that Lasts,Gary Chapman,4.8,25554,8,2018,Non Fiction
The 5 Love Languages: The Secret to Love that Lasts,Gary Chapman,4.8,25554,8,2019,Non Fiction
The 5000 Year Leap,W. Cleon Skousen,4.8,1680,12,2009,Non Fiction
The 7 Habits of Highly Effective People: Powerful Lessons in Personal Change,Stephen R. Covey,4.6,9325,24,2009,Non Fiction
The 7 Habits of Highly Effective People: Powerful Lessons in Personal Change,Stephen R. Covey,4.6,9325,24,2011,Non Fiction
The 7 Habits of Highly Effective People: Powerful Lessons in Personal Change,Stephen R. Covey,4.6,9325,24,2012,Non Fiction
The 7 Habits of Highly Effective People: Powerful Lessons in Personal Change,Stephen R. Covey,4.6,9325,24,2013,Non Fiction
The 7 Habits of Highly Effective People: Powerful Lessons in Personal Change,Stephen R. Covey,4.7,4725,16,2015,Non Fiction
The 7 Habits of Highly Effective People: Powerful Lessons in Personal Change,Stephen R. Covey,4.7,4725,16,2016,Non Fiction
The 7 Habits of Highly Effective People: Powerful Lessons in Personal Change,Stephen R. Covey,4.7,4725,16,2017,Non Fiction
The Alchemist,Paulo Coelho,4.7,35799,39,2014,Fiction
The Amateur,Edward Klein,4.6,2580,9,2012,Non Fiction
The Art of Racing in the Rain: A Novel,Garth Stein,4.7,11813,10,2010,Fiction
The Art of Racing in the Rain: A Novel,Garth Stein,4.7,11813,10,2011,Fiction
The Big Short: Inside the Doomsday Machine,Michael Lewis,4.7,3536,17,2010,Non Fiction
The Blood of Olympus (The Heroes of Olympus (5)),Rick Riordan,4.8,6600,11,2014,Fiction
"The Blood Sugar Solution: The UltraHealthy Program for Losing Weight, Preventing Disease, and Feeling Great Now!",Mark Hyman M.D.,4.2,1789,14,2012,Non Fiction
"The Body Keeps the Score: Brain, Mind, and Body in the Healing of Trauma",Bessel van der Kolk M.D.,4.8,12361,12,2019,Non Fiction
The Book of Basketball: The NBA According to The Sports Guy,Bill Simmons,4.7,858,53,2009,Non Fiction
The Book Thief,Markus Zusak,4.6,23148,6,2013,Fiction
The Book Thief,Markus Zusak,4.6,23148,6,2014,Fiction
The Book with No Pictures,B. J. Novak,4.8,8081,8,2014,Fiction
The Book with No Pictures,B. J. Novak,4.8,8081,8,2015,Fiction
The Boys in the Boat: Nine Americans and Their Epic Quest for Gold at the 1936 Berlin Olympics,Daniel James Brown,4.8,23358,12,2014,Non Fiction
The Boys in the Boat: Nine Americans and Their Epic Quest for Gold at the 1936 Berlin Olympics,Daniel James Brown,4.8,23358,12,2015,Non Fiction
The Casual Vacancy,J.K. Rowling,3.3,9372,12,2012,Fiction
The China Study: The Most Comprehensive Study of Nutrition Ever Conducted And the Startling Implications for Diet…,Thomas Campbell,4.7,4633,21,2011,Non Fiction
The Complete Ketogenic Diet for Beginners: Your Essential Guide to Living the Keto Lifestyle,Amy Ramos,4.3,13061,6,2018,Non Fiction
The Complete Ketogenic Diet for Beginners: Your Essential Guide to Living the Keto Lifestyle,Amy Ramos,4.3,13061,6,2019,Non Fiction
The Confession: A Novel,John Grisham,4.3,3523,13,2010,Fiction
The Constitution of the United States,Delegates of the Constitutional,4.8,2774,0,2016,Non Fiction
The Daily Show with Jon Stewart Presents Earth (The Book): A Visitor's Guide to the Human Race,Jon Stewart,4.4,440,11,2010,Non Fiction
The Day the Crayons Quit,Drew Daywalt,4.8,8922,9,2013,Fiction
The Day the Crayons Quit,Drew Daywalt,4.8,8922,9,2014,Fiction
The Day the Crayons Quit,Drew Daywalt,4.8,8922,9,2015,Fiction
"The Dukan Diet: 2 Steps to Lose the Weight, 2 Steps to Keep It Off Forever",Pierre Dukan,4.1,2023,15,2011,Non Fiction
The Elegance of the Hedgehog,Muriel Barbery,4,1859,11,2009,Fiction
The Fault in Our Stars,John Green,4.7,50482,13,2012,Fiction
The Fault in Our Stars,John Green,4.7,50482,13,2013,Fiction
The Fault in Our Stars,John Green,4.7,50482,7,2014,Fiction
The Fault in Our Stars,John Green,4.7,50482,13,2014,Fiction
The Five Dysfunctions of a Team: A Leadership Fable,Patrick Lencioni,4.6,3207,6,2009,Non Fiction
The Five Dysfunctions of a Team: A Leadership Fable,Patrick Lencioni,4.6,3207,6,2010,Non Fiction
The Five Dysfunctions of a Team: A Leadership Fable,Patrick Lencioni,4.6,3207,6,2011,Non Fiction
The Five Dysfunctions of a Team: A Leadership Fable,Patrick Lencioni,4.6,3207,6,2012,Non Fiction
The Five Dysfunctions of a Team: A Leadership Fable,Patrick Lencioni,4.6,3207,6,2013,Non Fiction
The Five Love Languages: How to Express Heartfelt Commitment to Your Mate,Gary Chapman,4.6,803,9,2009,Non Fiction
The Four Agreements: A Practical Guide to Personal Freedom (A Toltec Wisdom Book),Don Miguel Ruiz,4.7,23308,6,2013,Non Fiction
The Four Agreements: A Practical Guide to Personal Freedom (A Toltec Wisdom Book),Don Miguel Ruiz,4.7,23308,6,2015,Non Fiction
The Four Agreements: A Practical Guide to Personal Freedom (A Toltec Wisdom Book),Don Miguel Ruiz,4.7,23308,6,2016,Non Fiction
The Four Agreements: A Practical Guide to Personal Freedom (A Toltec Wisdom Book),Don Miguel Ruiz,4.7,23308,6,2017,Non Fiction
The Four Agreements: A Practical Guide to Personal Freedom (A Toltec Wisdom Book),Don Miguel Ruiz,4.7,23308,6,2018,Non Fiction
The Four Agreements: A Practical Guide to Personal Freedom (A Toltec Wisdom Book),Don Miguel Ruiz,4.7,23308,6,2019,Non Fiction
The Getaway,Jeff Kinney,4.8,5836,0,2017,Fiction
The Girl on the Train,Paula Hawkins,4.1,79446,18,2015,Fiction
The Girl on the Train,Paula Hawkins,4.1,79446,7,2016,Fiction
The Girl Who Kicked the Hornet's Nest (Millennium Trilogy),Stieg Larsson,4.7,7747,14,2010,Fiction
The Girl Who Kicked the Hornet's Nest (Millennium Trilogy),Stieg Larsson,4.7,7747,14,2011,Fiction
The Girl Who Played with Fire (Millennium Series),Stieg Larsson,4.7,7251,9,2010,Fiction
The Girl Who Played with Fire (Millennium),Stieg Larsson,4.7,7251,16,2009,Fiction
The Girl with the Dragon Tattoo (Millennium Series),Stieg Larsson,4.4,10559,2,2009,Fiction
The Girl with the Dragon Tattoo (Millennium Series),Stieg Larsson,4.4,10559,2,2010,Fiction
The Going-To-Bed Book,Sandra Boynton,4.8,5249,5,2016,Fiction
The Going-To-Bed Book,Sandra Boynton,4.8,5249,5,2017,Fiction
The Goldfinch: A Novel (Pulitzer Prize for Fiction),Donna Tartt,3.9,33844,20,2013,Fiction
The Goldfinch: A Novel (Pulitzer Prize for Fiction),Donna Tartt,3.9,33844,20,2014,Fiction
The Great Gatsby,F. Scott Fitzgerald,4.4,11616,7,2012,Fiction
The Great Gatsby,F. Scott Fitzgerald,4.4,11616,7,2013,Fiction
The Great Gatsby,F. Scott Fitzgerald,4.4,11616,7,2014,Fiction
The Guardians: A Novel,John Grisham,4.5,13609,14,2019,Fiction
The Guernsey Literary and Potato Peel Pie Society,Mary Ann Shaffer,4.7,8587,10,2009,Fiction
The Handmaid's Tale,Margaret Atwood,4.3,29442,7,2017,Fiction
The Harbinger: The Ancient Mystery that Holds the Secret of America's Future,Jonathan Cahn,4.6,11098,13,2012,Fiction
The Hate U Give,Angie Thomas,4.8,9947,11,2018,Fiction
The Help,Kathryn Stockett,4.8,13871,6,2009,Fiction
The Help,Kathryn Stockett,4.8,13871,6,2010,Fiction
The Help,Kathryn Stockett,4.8,13871,8,2011,Fiction
The Help,Kathryn Stockett,4.8,13871,7,2011,Fiction
"The House of Hades (Heroes of Olympus, Book 4)",Rick Riordan,4.8,6982,14,2013,Fiction
The Hunger Games,Suzanne Collins,4.7,32122,14,2010,Fiction
The Hunger Games (Book 1),Suzanne Collins,4.7,32122,8,2011,Fiction
The Hunger Games (Book 1),Suzanne Collins,4.7,32122,8,2012,Fiction
The Hunger Games Trilogy Boxed Set (1),Suzanne Collins,4.8,16949,30,2011,Fiction
The Hunger Games Trilogy Boxed Set (1),Suzanne Collins,4.8,16949,30,2012,Fiction
The Immortal Life of Henrietta Lacks,Rebecca Skloot,4.7,9289,13,2010,Non Fiction
The Immortal Life of Henrietta Lacks,Rebecca Skloot,4.7,9289,9,2011,Non Fiction
The Immortal Life of Henrietta Lacks,Rebecca Skloot,4.7,9289,9,2012,Non Fiction
The Instant Pot Electric Pressure Cooker Cookbook: Easy Recipes for Fast & Healthy Meals,Laurel Randolph,4.3,7368,7,2017,Non Fiction
The Instant Pot Electric Pressure Cooker Cookbook: Easy Recipes for Fast & Healthy Meals,Laurel Randolph,4.3,7368,7,2018,Non Fiction
The Last Lecture,Randy Pausch,4.7,4028,9,2009,Non Fiction
"The Last Olympian (Percy Jackson and the Olympians, Book 5)",Rick Riordan,4.8,4628,7,2009,Fiction
"The Last Olympian (Percy Jackson and the Olympians, Book 5)",Rick Riordan,4.8,4628,7,2010,Fiction
The Legend of Zelda: Hyrule Historia,Patrick Thorpe,4.9,5396,20,2013,Fiction
The Lego Ideas Book: Unlock Your Imagination,Daniel Lipkowitz,4.4,4247,13,2011,Non Fiction
The Lego Ideas Book: Unlock Your Imagination,Daniel Lipkowitz,4.4,4247,13,2012,Non Fiction
The Life-Changing Magic of Tidying Up: The Japanese Art of Decluttering and Organizing,Marie Kondō,4.5,22641,11,2015,Non Fiction
The Life-Changing Magic of Tidying Up: The Japanese Art of Decluttering and Organizing,Marie Kondō,4.5,22641,11,2016,Non Fiction
The Life-Changing Magic of Tidying Up: The Japanese Art of Decluttering and Organizing,Marie Kondō,4.5,22641,11,2017,Non Fiction
The Life-Changing Magic of Tidying Up: The Japanese Art of Decluttering and Organizing,Marie Kondō,4.5,22641,11,2019,Non Fiction
The Litigators,John Grisham,4.4,6222,18,2011,Fiction
"The Lost Hero (Heroes of Olympus, Book 1)",Rick Riordan,4.8,4506,14,2010,Fiction
The Lost Symbol,Dan Brown,4.2,8747,19,2009,Fiction
The Love Dare,Stephen Kendrick,4.8,1655,13,2009,Non Fiction
The Magnolia Story,Chip Gaines,4.9,7861,5,2016,Non Fiction
"The Mark of Athena (Heroes of Olympus, Book 3)",Rick Riordan,4.8,6247,10,2012,Fiction
The Martian,Andy Weir,4.7,39459,9,2015,Fiction
The Maze Runner (Book 1),James Dashner,4.5,10101,8,2014,Fiction
The Meltdown (Diary of a Wimpy Kid Book 13),Jeff Kinney,4.8,5898,8,2018,Fiction
The Mueller Report,The Washington Post,4.6,2744,12,2019,Non Fiction
The Nightingale: A Novel,Kristin Hannah,4.8,49288,11,2015,Fiction
The Nightingale: A Novel,Kristin Hannah,4.8,49288,11,2016,Fiction
The Official SAT Study Guide,The College Board,4.4,1201,40,2010,Non Fiction
The Official SAT Study Guide,The College Board,4.4,1201,40,2011,Non Fiction
The Official SAT Study Guide,The College Board,4.4,1201,40,2012,Non Fiction
The Official SAT Study Guide,The College Board,4.4,1201,40,2013,Non Fiction
The Official SAT Study Guide,The College Board,4.4,1201,40,2014,Non Fiction
"The Official SAT Study Guide, 2016 Edition (Official Study Guide for the New Sat)",The College Board,4.3,807,36,2016,Non Fiction
The Paris Wife: A Novel,Paula McLain,4.3,3759,16,2011,Fiction
"The Pioneer Woman Cooks: A Year of Holidays: 140 Step-by-Step Recipes for Simple, Scrumptious Celebrations",Ree Drummond,4.8,2663,17,2013,Non Fiction
"The Pioneer Woman Cooks: Dinnertime - Comfort Classics, Freezer Food, 16-minute Meals, and Other Delicious Ways to Solve…",Ree Drummond,4.8,3428,14,2015,Non Fiction
The Pioneer Woman Cooks: Food from My Frontier,Ree Drummond,4.8,2876,21,2012,Non Fiction
"The Plant Paradox Cookbook: 100 Delicious Recipes to Help You Lose Weight, Heal Your Gut, and Live Lectin-Free",Dr. Steven R Gundry MD,4.5,3601,18,2018,Non Fiction
"The Plant Paradox: The Hidden Dangers in ""Healthy"" Foods That Cause Disease and Weight Gain",Dr. Steven R Gundry MD,4.4,7058,17,2018,Non Fiction
The Pout-Pout Fish,Deborah Diesen,4.8,9784,5,2017,Fiction
The Pout-Pout Fish,Deborah Diesen,4.8,9784,5,2018,Fiction
The Power of Habit: Why We Do What We Do in Life and Business,Charles Duhigg,4.6,10795,21,2012,Non Fiction
The President Is Missing: A Novel,James Patterson,4.3,10191,18,2018,Fiction
The Racketeer,John Grisham,4.3,14493,18,2012,Fiction
"The Red Pyramid (The Kane Chronicles, Book 1)",Rick Riordan,4.6,2186,12,2010,Fiction
"The Road to Serfdom: Text and Documents--The Definitive Edition (The Collected Works of F. A. Hayek, Volume 2)",F. A. Hayek,4.6,1204,14,2010,Non Fiction
"The Serpent's Shadow (The Kane Chronicles, Book 3)",Rick Riordan,4.8,2091,12,2012,Fiction
The Shack: Where Tragedy Confronts Eternity,William P. Young,4.6,19720,8,2009,Fiction
The Shack: Where Tragedy Confronts Eternity,William P. Young,4.6,19720,8,2017,Fiction
The Short Second Life of Bree Tanner: An Eclipse Novella (The Twilight Saga),Stephenie Meyer,4.6,2122,0,2010,Fiction
The Silent Patient,Alex Michaelides,4.5,27536,14,2019,Fiction
"The Son of Neptune (Heroes of Olympus, Book 2)",Rick Riordan,4.8,4290,10,2011,Fiction
The Subtle Art of Not Giving a F*ck: A Counterintuitive Approach to Living a Good Life,Mark Manson,4.6,26490,15,2017,Non Fiction
The Subtle Art of Not Giving a F*ck: A Counterintuitive Approach to Living a Good Life,Mark Manson,4.6,26490,15,2018,Non Fiction
The Subtle Art of Not Giving a F*ck: A Counterintuitive Approach to Living a Good Life,Mark Manson,4.6,26490,15,2019,Non Fiction
The Sun and Her Flowers,Rupi Kaur,4.7,5487,9,2017,Non Fiction
"The Third Wheel (Diary of a Wimpy Kid, Book 7)",Jeff Kinney,4.7,6377,7,2012,Fiction
"The Throne of Fire (The Kane Chronicles, Book 2)",Rick Riordan,4.7,1463,10,2011,Fiction
The Time Traveler's Wife,Audrey Niffenegger,4.4,3759,6,2009,Fiction
The Tipping Point: How Little Things Can Make a Big Difference,Malcolm Gladwell,4.4,3503,9,2009,Non Fiction
The Total Money Makeover: Classic Edition: A Proven Plan for Financial Fitness,Dave Ramsey,4.7,11550,10,2019,Non Fiction
The Twilight Saga Collection,Stephenie Meyer,4.7,3801,82,2009,Fiction
"The Ugly Truth (Diary of a Wimpy Kid, Book 5)",Jeff Kinney,4.8,3796,12,2010,Fiction
The Unofficial Harry Potter Cookbook: From Cauldron Cakes to Knickerbocker Glory--More Than 150 Magical Recipes for…,Dinah Bucholz,4.7,9030,10,2019,Non Fiction
The Very Hungry Caterpillar,Eric Carle,4.9,19546,5,2013,Fiction
The Very Hungry Caterpillar,Eric Carle,4.9,19546,5,2014,Fiction
The Very Hungry Caterpillar,Eric Carle,4.9,19546,5,2015,Fiction
The Very Hungry Caterpillar,Eric Carle,4.9,19546,5,2016,Fiction
The Very Hungry Caterpillar,Eric Carle,4.9,19546,5,2017,Fiction
The Very Hungry Caterpillar,Eric Carle,4.9,19546,5,2018,Fiction
The Very Hungry Caterpillar,Eric Carle,4.9,19546,5,2019,Fiction
The Whole30: The 30-Day Guide to Total Health and Food Freedom,Melissa Hartwig Urban,4.6,7508,16,2015,Non Fiction
The Whole30: The 30-Day Guide to Total Health and Food Freedom,Melissa Hartwig Urban,4.6,7508,16,2016,Non Fiction
The Whole30: The 30-Day Guide to Total Health and Food Freedom,Melissa Hartwig Urban,4.6,7508,16,2017,Non Fiction
The Wonderful Things You Will Be,Emily Winfield Martin,4.9,8842,10,2016,Fiction
The Wonderful Things You Will Be,Emily Winfield Martin,4.9,8842,10,2017,Fiction
The Wonderful Things You Will Be,Emily Winfield Martin,4.9,8842,10,2018,Fiction
The Wonderful Things You Will Be,Emily Winfield Martin,4.9,8842,10,2019,Fiction
The Wonky Donkey,Craig Smith,4.8,30183,4,2018,Fiction
The Wonky Donkey,Craig Smith,4.8,30183,4,2019,Fiction
The Wright Brothers,David McCullough,4.7,6169,16,2015,Non Fiction
"Things That Matter: Three Decades of Passions, Pastimes and Politics [Deckled Edge]",Charles Krauthammer,4.7,7034,15,2013,Non Fiction
"Thinking, Fast and Slow",Daniel Kahneman,4.6,11034,19,2011,Non Fiction
"Thinking, Fast and Slow",Daniel Kahneman,4.6,11034,19,2012,Non Fiction
Thirteen Reasons Why,Jay Asher,4.5,7932,9,2017,Fiction
Thomas Jefferson: The Art of Power,Jon Meacham,4.5,1904,23,2012,Non Fiction
Three Cups of Tea: One Man's Mission to Promote Peace - One School at a Time,Greg Mortenson,4.3,3319,11,2009,Non Fiction
Three Cups of Tea: One Man's Mission to Promote Peace - One School at a Time,Greg Mortenson,4.3,3319,11,2010,Non Fiction
Thug Kitchen: The Official Cookbook: Eat Like You Give a F*ck (Thug Kitchen Cookbooks),Thug Kitchen,4.6,11128,23,2014,Non Fiction
Thug Kitchen: The Official Cookbook: Eat Like You Give a F*ck (Thug Kitchen Cookbooks),Thug Kitchen,4.6,11128,23,2015,Non Fiction
Thug Kitchen: The Official Cookbook: Eat Like You Give a F*ck (Thug Kitchen Cookbooks),Thug Kitchen,4.6,11128,23,2016,Non Fiction
Thug Kitchen: The Official Cookbook: Eat Like You Give a F*ck (Thug Kitchen Cookbooks),Thug Kitchen,4.6,11128,23,2017,Non Fiction
Tina Fey: Bossypants,Tina Fey,4.3,5977,12,2011,Non Fiction
To Kill a Mockingbird,Harper Lee,4.8,26234,0,2013,Fiction
To Kill a Mockingbird,Harper Lee,4.8,26234,0,2014,Fiction
To Kill a Mockingbird,Harper Lee,4.8,26234,0,2015,Fiction
To Kill a Mockingbird,Harper Lee,4.8,26234,0,2016,Fiction
To Kill a Mockingbird,Harper Lee,4.8,26234,7,2019,Fiction
"Tools of Titans: The Tactics, Routines, and Habits of Billionaires, Icons, and World-Class Performers",Timothy Ferriss,4.6,4360,21,2017,Non Fiction
"Towers of Midnight (Wheel of Time, Book Thirteen)",Robert Jordan,4.8,2282,21,2010,Fiction
True Compass: A Memoir,Edward M. Kennedy,4.5,438,15,2009,Non Fiction
"Twilight (The Twilight Saga, Book 1)",Stephenie Meyer,4.7,11676,9,2009,Fiction
Ultimate Sticker Book: Frozen: More Than 60 Reusable Full-Color Stickers,DK,4.5,2586,5,2014,Fiction
"Unbroken: A World War II Story of Survival, Resilience, and Redemption",Laura Hillenbrand,4.8,29673,16,2010,Non Fiction
"Unbroken: A World War II Story of Survival, Resilience, and Redemption",Laura Hillenbrand,4.8,29673,16,2011,Non Fiction
"Unbroken: A World War II Story of Survival, Resilience, and Redemption",Laura Hillenbrand,4.8,29673,16,2012,Non Fiction
"Unbroken: A World War II Story of Survival, Resilience, and Redemption",Laura Hillenbrand,4.8,29673,13,2014,Non Fiction
"Unbroken: A World War II Story of Survival, Resilience, and Redemption",Laura Hillenbrand,4.8,29673,16,2014,Non Fiction
Under the Dome: A Novel,Stephen King,4.3,6740,20,2009,Fiction
Unfreedom of the Press,Mark R. Levin,4.9,5956,11,2019,Non Fiction
Unicorn Coloring Book: For Kids Ages 4-8 (US Edition) (Silly Bear Coloring Books),Silly Bear,4.8,6108,4,2019,Non Fiction
"Uninvited: Living Loved When You Feel Less Than, Left Out, and Lonely",Lysa TerKeurst,4.7,4585,9,2016,Non Fiction
Watchmen,Alan Moore,4.8,3829,42,2009,Fiction
Water for Elephants: A Novel,Sara Gruen,4.5,8958,12,2011,Fiction
What Happened,Hillary Rodham Clinton,4.6,5492,18,2017,Non Fiction
What If?: Serious Scientific Answers to Absurd Hypothetical Questions,Randall Munroe,4.7,9292,17,2014,Non Fiction
What Pet Should I Get? (Classic Seuss),Dr. Seuss,4.7,1873,14,2015,Fiction
What Should Danny Do? (The Power to Choose Series),Adir Levy,4.8,8170,13,2019,Fiction
What to Expect When You're Expecting,Heidi Murkoff,4.4,3341,9,2011,Non Fiction
"Wheat Belly: Lose the Wheat, Lose the Weight, and Find Your Path Back to Health",William Davis,4.4,7497,6,2012,Non Fiction
"Wheat Belly: Lose the Wheat, Lose the Weight, and Find Your Path Back to Health",William Davis,4.4,7497,6,2013,Non Fiction
When Breath Becomes Air,Paul Kalanithi,4.8,13779,14,2016,Non Fiction
Where the Crawdads Sing,Delia Owens,4.8,87841,15,2019,Fiction
Where the Wild Things Are,Maurice Sendak,4.8,9967,13,2009,Fiction
Whose Boat Is This Boat?: Comments That Don't Help in the Aftermath of a Hurricane,The Staff of The Late Show with,4.6,6669,12,2018,Non Fiction
Wild: From Lost to Found on the Pacific Crest Trail,Cheryl Strayed,4.4,17044,18,2012,Non Fiction
Winter of the World: Book Two of the Century Trilogy,Ken Follett,4.5,10760,15,2012,Fiction
Women Food and God: An Unexpected Path to Almost Everything,Geneen Roth,4.2,1302,11,2010,Non Fiction
Wonder,R. J. Palacio,4.8,21625,9,2013,Fiction
Wonder,R. J. Palacio,4.8,21625,9,2014,Fiction
Wonder,R. J. Palacio,4.8,21625,9,2015,Fiction
Wonder,R. J. Palacio,4.8,21625,9,2016,Fiction
Wonder,R. J. Palacio,4.8,21625,9,2017,Fiction
Wrecking Ball (Diary of a Wimpy Kid Book 14),Jeff Kinney,4.9,9413,8,2019,Fiction
You Are a Badass: How to Stop Doubting Your Greatness and Start Living an Awesome Life,Jen Sincero,4.7,14331,8,2016,Non Fiction
You Are a Badass: How to Stop Doubting Your Greatness and Start Living an Awesome Life,Jen Sincero,4.7,14331,8,2017,Non Fiction
You Are a Badass: How to Stop Doubting Your Greatness and Start Living an Awesome Life,Jen Sincero,4.7,14331,8,2018,Non Fiction
You Are a Badass: How to Stop Doubting Your Greatness and Start Living an Awesome Life,Jen Sincero,4.7,14331,8,2019,Non Fiction
g bestsellers.with.categories.csv…]()


# Output-
![image](https://github.com/user-attachments/assets/6a467747-eafe-4fea-b11c-4506857f7131)
