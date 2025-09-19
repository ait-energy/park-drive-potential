package at.ac.ait.matsim.salabim.runners;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.locationtech.jts.io.ParseException;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.api.core.v01.population.Population;
import org.matsim.api.core.v01.population.Route;
import org.matsim.core.population.PersonUtils;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.population.routes.NetworkRoute;
import org.matsim.core.router.TripStructureUtils;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.core.utils.misc.OptionalTime;

import com.google.common.base.Joiner;

/**
 * Export all trips from a MATSim population to a CSV file.
 * Add information about subtours to each trip, i.e.
 * a unique subtour id per person, the total number of trips in the subtour,
 * and the number of the trip in the subtour.
 *
 * subtourNr and tripNr are 1-based.
 *
 * First developed for project INTRO.
 */
public class PopulationExtractSubTours {

	private static final Logger LOGGER = LogManager.getLogger();

	private final String activityType;
	private int tripCount;
	private int subtourCount;

	public PopulationExtractSubTours(String activityType) {
		this.activityType = activityType;
	}

	public void run(String populationXml, Path tripChainsCsv) throws ParseException, IOException {
		Population population = PopulationUtils.readPopulation(populationXml);
		LOGGER.info("read {} agents.", population.getPersons().size());

		tripCount = 0;
		subtourCount = 0;
		try (BufferedWriter writer = Files.newBufferedWriter(tripChainsCsv)) {
			writer.write("""
					personId,age,sex,isEmployed,fullTimeWork,highEducation,highIncome,\
					subtourNr,subtourLen,tripNr,mode,departureSecondsOfDay,arrivalSecondsOfDay,\
					nextActivityStartSecondsOfDay,activityChain,links
					""");

			for (Person person : population.getPersons().values()) {
				writeSubtoursForPerson(person, writer);
			}
		}
		LOGGER.info("wrote {} trips in {} subtours for {} agents.", tripCount, subtourCount,
				population.getPersons().size());
	}

	private void writeSubtoursForPerson(Person person, BufferedWriter writer)
			throws IOException {
		List<PlanElement> planElements = person.getSelectedPlan().getPlanElements();
		List<Integer> subtourIndices = getActivityIndicesPlusFirstAndLast(planElements, activityType);
		for (int subtourNr = 0; subtourNr < subtourIndices.size() - 1; subtourNr++) {
			List<PlanElement> subTour = planElements.subList(
					subtourIndices.get(subtourNr),
					subtourIndices.get(subtourNr + 1) + 1);

			List<Trip> trips = TripStructureUtils.getTrips(subTour);
			subtourCount++;
			tripCount += trips.size();

			for (int tripNr = 0; tripNr < trips.size(); tripNr++) {
				Trip trip = trips.get(tripNr);
				List<String> cols = new ArrayList<>();

				cols.add(person.getId().toString());
				cols.add("%d".formatted(PersonUtils.getAge(person)));
				cols.add(PersonUtils.getSex(person));
				cols.add(PersonUtils.isEmployed(person) ? "1" : "0");
				cols.add(((Boolean) person.getAttributes().getAttribute("fullTimeWork") ? "1" : "0"));
				cols.add(((Boolean) person.getAttributes().getAttribute("highEducation") ? "1" : "0"));
				cols.add(((Boolean) person.getAttributes().getAttribute("highIncome") ? "1" : "0"));
				cols.add("%d".formatted(subtourNr + 1));
				cols.add("%d".formatted(trips.size()));
				cols.add("%d".formatted(tripNr + 1));
				String mode = trip.getLegsOnly().get(0).getRoutingMode();
				cols.add(mode);
				String departureTime = "%.0f".formatted(trip.getOriginActivity().getEndTime().orElse(-1));
				cols.add(departureTime);
				String arrivalTime = "%.0f".formatted(getArrivalTime(trip).orElse(-1));
				cols.add(arrivalTime);
				String nextActivityStartTime = "%.0f"
						.formatted(trip.getDestinationActivity().getStartTime().orElse(-1));
				cols.add(nextActivityStartTime);
				String activityChain = trip.getOriginActivity().getType() + "-"
						+ trip.getDestinationActivity().getType();
				cols.add(activityChain);
				String links = "\"%s\"".formatted(Joiner.on(",").join(getAllNetworkLinks(trip)));
				cols.add(links);
				writer.write(Joiner.on(",").join(cols) + "\n");
			}
		}
	}

	/**
	 * Extract the arrival time from departure time plus travel time
	 * (or simply take the start time of the destination activity as fallback in
	 * case not travel time is defined).
	 */
	private OptionalTime getArrivalTime(Trip trip) {
		OptionalTime tripStartTime = trip.getOriginActivity().getEndTime();
		OptionalTime fallbackArrivalTime = trip.getDestinationActivity().getStartTime();

		if (tripStartTime.isUndefined()) {
			return fallbackArrivalTime;
		}

		double arrivalTime = tripStartTime.orElse(0);
		for (Leg leg : trip.getLegsOnly()) {
			if (leg.getTravelTime().isUndefined()) {
				return fallbackArrivalTime;
			}
			arrivalTime += leg.getTravelTime().orElse(0);
		}
		return OptionalTime.defined(arrivalTime);
	}

	private static List<Id<Link>> getAllNetworkLinks(Trip trip) {
		List<Id<Link>> linkIds = new ArrayList<>();
		for (Leg leg : trip.getLegsOnly()) {
			Route route = leg.getRoute();
			if (route != null && route instanceof NetworkRoute) {
				linkIds.addAll(((NetworkRoute) route).getLinkIds());
			}
		}
		return linkIds;
	}

	private static List<Integer> getActivityIndicesPlusFirstAndLast(List<PlanElement> elements, String activityType) {
		Set<Integer> relevantIndices = elements.stream()
				.filter(e -> e instanceof Activity)
				.filter(e -> ((Activity) e).getType().equals(activityType))
				.map(e -> elements.indexOf(e))
				.collect(Collectors.toSet());
		relevantIndices = new HashSet<>(relevantIndices);

		relevantIndices.add(0);
		relevantIndices.add(elements.size() - 1);

		List<Integer> sortedIndices = new ArrayList<>(relevantIndices);
		Collections.sort(sortedIndices);
		return sortedIndices;
	}

	public static void main(String[] args) throws ParseException, IOException {
		if (args.length != 3) {
			LOGGER.error(
					"Exactly three arguments expected: input population file, activity type, output csv file.");
			return;
		}

		String populationXml = args[0];
		String activityType = args[1];
		Path tripChainsCsv = Paths.get(args[2]);

		if (Files.exists(tripChainsCsv)) {
			LOGGER.error("output file {} already exists.", tripChainsCsv);
			return;
		}
		new PopulationExtractSubTours(activityType).run(populationXml, tripChainsCsv);
	}

}
