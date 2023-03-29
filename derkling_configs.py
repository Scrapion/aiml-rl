from random import sample
from rewards.dps import reward_dps
from rewards.tank import reward_tank
from rewards.healer import reward_healer

dps_items = [["Blaster", "Talons", "BloodClaws", "Magnum", "Pistol"], ["HeliumBubblegum", "FrogLegs"], ["VampireGland", "ParalyzingDart"]]
healer_items = [["Magnum", "Pistol", "BloodClaws"], ["IronBubblegum", "HeliumBubblegum"], ["HealingGland"]]
tank_items = [["BloodClaws","Cleavers","Cripplers"], ["Shell", "Trombone", "IronBubblegum"], ["ParalyzingDart"]]


def get_home():
    home_team = [
        {
            'primaryColor': '#33ff33',
            'slots': sample(dps_items[0],1)+sample(dps_items[1],1)+sample(dps_items[2],1),
            'rewardFunction':reward_dps
        },{
            'primaryColor': '#009900',
            'slots': sample(healer_items[0],1)+sample(healer_items[1],1)+sample(healer_items[2],1),
            'rewardFunction':reward_healer
        },{
            'primaryColor': '#003300',
            'slots': sample(tank_items[0],1)+sample(tank_items[1],1)+sample(tank_items[2],1),
            'rewardFunction':reward_tank
        }
    ]
    return home_team

def get_away():
    away_team = [
        {
            'primaryColor': '#FF9999',
            'slots': sample(dps_items[0],1)+sample(dps_items[1],1)+sample(dps_items[2],1),
            'rewardFunction':reward_dps
        },{
            'primaryColor': '#CC0000',
            'slots': sample(healer_items[0],1)+sample(healer_items[1],1)+sample(healer_items[2],1),
            'rewardFunction':reward_healer
        },{
            'primaryColor': '#330000',
            'slots': sample(tank_items[0],1)+sample(tank_items[1],1)+sample(tank_items[2],1),
            'rewardFunction':reward_tank
        }
    ]
    return away_team


def get_config(n_arenas):
    away_team = []
    home_team = []
    for _ in range(n_arenas):
        away_team+=get_away()
        home_team+=get_home()

    return home_team, away_team

def get_config_fixed(n_arenas):
    away_team=[
    {
        'primaryColor': '#FF9999',
        'slots': ["Blaster","FrogLegs","VampireGland"],
        'rewardFunction':reward_dps
    },{
        'primaryColor': '#CC0000',
        'slots': ["Magnum","IronBubblegum","HealingGland"],
        'rewardFunction':reward_healer
    },{
        'primaryColor': '#330000',
        'slots': ["BloodClaws","Trombone","ParalyzingDart"],
        'rewardFunction':reward_tank
    }
    ]
    home_team=[
    {
        'primaryColor': '#33ff33',
        'slots': ["Blaster","FrogLegs","VampireGland"],
        'rewardFunction':reward_dps
    },{
        'primaryColor': '#009900',
        'slots': ["Magnum","IronBubblegum","HealingGland"],
        'rewardFunction':reward_healer
    },{
        'primaryColor': '#003300',
        'slots': ["BloodClaws","Trombone","ParalizingDart"],
        'rewardFunction':reward_tank
    }
    ]

    return home_team, away_team