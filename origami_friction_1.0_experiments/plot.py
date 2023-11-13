"""Plot fitness over generations for all experiments, averaged."""

import config
import matplotlib.pyplot as plt
import pandas
from experiment import Experiment
from generation import Generation
from individual import Individual
from population import Population
from revolve2.ci_group.logging import setup_logging
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from sqlalchemy import select


def main() -> None:
    """Run the program."""
    setup_logging()

    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    df = pandas.read_sql(
        select(
            Experiment.id.label("experiment_id"),
            Generation.generation_index,
            Individual.fitness,
        )
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id),
        dbengine,
    )

    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({"fitness": ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        "max_fitness",
        "mean_fitness",
    ]

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({"max_fitness": ["mean", "std"], "mean_fitness": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index",
        "max_fitness_mean",
        "max_fitness_std",
        "mean_fitness_mean",
        "mean_fitness_std",
    ]


  

    # Step 1: Determine the maximum fitness value
    max_fitness_value = df["fitness"].max()

    # Step 2: Normalize the fitness values
    agg_per_experiment_per_generation["max_fitness_normalized"] = agg_per_experiment_per_generation["max_fitness"] / max_fitness_value
    agg_per_experiment_per_generation["mean_fitness_normalized"] = agg_per_experiment_per_generation["mean_fitness"] / max_fitness_value

    # Step 3: Update the aggregation for plotting
    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({
            "max_fitness_normalized": ["mean", "std"],
            "mean_fitness_normalized": ["mean", "std"]
        })
        .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index",
        "max_fitness_mean_normalized",
        "max_fitness_std_normalized",
        "mean_fitness_mean_normalized",
        "mean_fitness_std_normalized",
    ]

    #Statitsstics
    # Calculate additional statistics for normalized mean and max fitness
    stats = agg_per_generation.agg({
        "max_fitness_mean_normalized": ["mean", "std", "median"],
        "mean_fitness_mean_normalized": ["mean", "std", "median"]
    }).reset_index()

    # Flatten the column headers
    stats.columns = ["statistic", "max_fitness_mean_normalized", "mean_fitness_mean_normalized"]

    # Print the statistics to console
    print(stats)

    # Save the statistics to a CSV file
    stats.to_csv("normalized_fitness_statistics.csv", index=False)



    plt.figure()

    # Plot max
    plt.plot(
    agg_per_generation["generation_index"],
    agg_per_generation["max_fitness_mean_normalized"],
    label="Max fitness (normalized)",
    color="b",
)
    
    plt.fill_between(
    agg_per_generation["generation_index"],
    agg_per_generation["max_fitness_mean_normalized"] - agg_per_generation["max_fitness_std_normalized"],
    agg_per_generation["max_fitness_mean_normalized"] + agg_per_generation["max_fitness_std_normalized"],
    color="b",
    alpha=0.2,
)
    plt.plot(
    agg_per_generation["generation_index"],
    agg_per_generation["mean_fitness_mean_normalized"],
    label="Mean fitness (normalized)",
    color="r",
)
    plt.fill_between(
    agg_per_generation["generation_index"],
    agg_per_generation["mean_fitness_mean_normalized"] - agg_per_generation["mean_fitness_std_normalized"],
    agg_per_generation["mean_fitness_mean_normalized"] + agg_per_generation["mean_fitness_std_normalized"],
    color="r",
    alpha=0.2,
)

    

    plt.xlabel("Generation index")
    plt.ylabel("Fitness")
    plt.title("Mean and max fitness across repetitions with std as shade")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
