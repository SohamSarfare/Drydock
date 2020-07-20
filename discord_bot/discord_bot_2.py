import discord
from discord.ext import commands
import config
import os
import traceback
import random 
import pandas as pd

bot = commands.Bot(command_prefix= '!')

@bot.event
async def on_ready():
    print('Ready to roll')

@bot.command(aliases = ['Hello', 'hellothere'])
async def hello(ctx):
    """Greets you with a warm message"""
    await ctx.send('General Kenobi')

@bot.command()
async def photo(ctx):
    """Sends random Photo"""
    entries = os.scandir('E:\\Downloads\\wallpapers\\Final')
    photos = []
    for entry in entries:
        photos.append(entry.name)
    photo_upload = random.choices(photos)[0]
    file = discord.File(r'E:\Downloads\wallpapers\Final\{}'.format(photo_upload), photo_upload)
    await ctx.send(file = file)

@bot.command(aliases = ['answered'])
async def archive(ctx):
    """Compiles an archive of all questions answered in previous drydock videos 
    Aliases: \'archive\' \'answered\'"""
    try:
        await ctx.send("Processing, gimme a minute...")
        import question_get
        file = discord.File(r'files\Drydock Questions.xlsx', 'Drydock Questions.xlsx')
        await ctx.send(file = file)
    except:
        file = discord.File(r'files\Drydock Questions.xlsx', 'Drydock Questions.xlsx')
        await ctx.send(file = file)

@bot.command()
async def getquestions(ctx,*,video_id):
    """Scraps videos to gain all question replies made to \'Pinned Post for Q&A Comment\'
    Usage:
    Get questions asked on all videos- !getquestions all
    Get questions asked on a certain video: !getquestions <video url>"""
    import time
    start = time.time()
    import re
    import question_scrapper
    import question_classifier
    try:
        reg = re.compile(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?(?P<id>[A-Za-z0-9\-=_]{11})')
        video_id = ''
        check = True
        try:
            video_id = reg.match(video_id).group('id')
            check = question_scrapper.check(video_id)
        except AttributeError:
            video_id = 'all'

        if video_id == 'all' and check:
            await ctx.send("This may take a while, hang on...")
            question_scrapper.lastVideo()

            await ctx.send("Scrapping done, classifying now...")
            data = pd.read_excel(r'files\questions.xlsx')
            acc = question_classifier.classify(data, 'all')

            file = discord.File(r'files\questions_classified_svc.xlsx', 'Drydock Questions.xlsx')
            await ctx.send(file = file)
            await ctx.send("Accuracy of classification: {}".format(acc))

            end = time.time()
            await ctx.send("Time taken: {} seconds".format(end- start))
        elif check:
            await ctx.send("This may take a while, hang on...")
            question_scrapper.scrapVideo(video_id)

            await ctx.send("Scrapping done, classifying now...")
            data = pd.read_excel(r"files\questions_{}.xlsx".format(video_id))
            acc = question_classifier.classify(data, video_id)

            file = discord.File(r"files\questions_classified_{}.xlsx".format(video_id), "Questions_{}.xlsx".format(video_id))
            await ctx.send(file = file)
            await ctx.send("Accuracy of classification: {}".format(acc))

            #os.remove(r"files\questions_{}.xlsx".format(video_id))
            #os.remove(r"files\questions_classified_{}.xlsx".format(video_id))

            end = time.time()
            await ctx.send("Time taken: {} seconds".format(end- start))
        else:
            await ctx.send("Currently the bot does not support this youtube channel")
    
    except AttributeError as err:        
        traceback.print_exc()
        print(err)
        await ctx.send("Some error occured")

    except PermissionError as err:
        traceback.print_exc()   
        print(err)
        await ctx.send("Some error occured")
     
    except OSError as err:
        traceback.print_exc()
        print(err)
        await ctx.send("Some error occured")
     
    except ValueError as err:
        traceback.print_exc()        
        print(err)
        await ctx.send("There are no questions made on this video")
        os.remove(r"files\questions_{}.xlsx".format(video_id))
        os.remove(r"files\questions_classified_{}.xlsx".format(video_id))

    except KeyError as err:
        traceback.print_exc()
        print(err)
        await ctx.send("Some error occured")

bot.run(config.credentials['bot_id'])